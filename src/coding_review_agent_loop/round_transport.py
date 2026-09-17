"""Bounded, dependency-leaf transport for durable round comments."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import zlib
from collections.abc import Mapping, Sequence

from .errors import AgentLoopError
from .protocol_markers import TrustedBody, sanitize_historical_text, scan_reserved_markers

MAX_GITHUB_BODY_CHARS = 60_000
MAX_PLAN_VALIDATION_DIAGNOSTIC_CHARS = 4096
ROUND_RESUME_MARKER_RE = re.compile(
    r"<!--\s*AGENT_LOOP_META:\s*(?P<payload>[A-Za-z0-9+/=_:-]+)\s*-->", re.I
)
ROUND_TRANSPORT_SIDECAR_RE = re.compile(
    r"<!--\s*AGENT_LOOP_SIDECAR:\s*(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->", re.I
)
PLAN_VALIDATION_DIAGNOSTIC_MARKER_RE = re.compile(
    r"<!--\s*AGENT_PLAN_VALIDATION_DIAGNOSTIC:\s*"
    r"(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->",
    re.I,
)
_EXECUTION_RECOMMENDATION_RE = re.compile(
    r"<!--\s*AGENT_EXECUTION_RECOMMENDATION:\s*"
    r"(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->",
    re.I,
)
_EXECUTION_RECOMMENDATION_SECTION_BOUNDARY_RE = re.compile(
    r"(?m)^<!--\s*execution-recommendation-section:\s*"
    r"(?P<digest>[0-9a-f]{64})\s*-->\r?$",
    re.I,
)
_RISK_TEST_MATRIX_RE = re.compile(
    r"<!--\s*AGENT_RISK_TEST_MATRIX:\s*(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->",
    re.I,
)
_RISK_TEST_MATRIX_SECTION_BOUNDARY_RE = re.compile(
    r"(?m)^<!--\s*risk-test-matrix-section:\s*(?P<identity>[0-9a-f]{64})\s*-->\r?$",
    re.I,
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
    "local_test_evidence",
    "execution_recommendation",
    "risk_test_matrix_payload",
    "risk_test_matrix_changes_payload",
    "risk_test_matrix_marker",
)
_MAX_COMPRESSED = 8_000_000
_MAX_DECOMPRESSED = 16_000_000
_PART_CHARS = 40_000
_RISK_MATRIX_COMPACT_AT_CHARS = 50_000
_RISK_MATRIX_MARKER_COMPACT_AT_CHARS = 4_000


def execution_recommendation_section_boundary(encoded: str) -> str:
    """Return the renderer-owned boundary for one recommendation marker."""
    digest = hashlib.sha256(encoded.encode("ascii")).hexdigest()
    return f"<!-- execution-recommendation-section: {digest} -->"


def risk_test_matrix_section_boundary(identity: str) -> str:
    """Return a boundary bound to the authenticated structured matrix identity.

    This marker authenticates which structured payload owns the rendered
    section. Recovery checks the identity only; it never requires historical
    Markdown to be byte-stable across renderer versions.
    """
    if not re.fullmatch(r"[0-9a-f]{64}", identity):
        raise AgentLoopError("Risk matrix section boundary requires a SHA-256 identity.")
    return f"<!-- risk-test-matrix-section: {identity} -->"


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


def _prepare_execution_recommendation_transport(
    body_text: str,
) -> tuple[str, list[TrustedBody], tuple[int, int, str] | None]:
    """Spill an oversized v1 recommendation into bounded round sidecars.

    The recommendation marker remains in the public anchor as a small reference;
    the complete canonical JSON is carried losslessly by ordinary round sidecars.
    This keeps large scope ledgers from being duplicated in the visible comment.
    """
    matches = list(_EXECUTION_RECOMMENDATION_RE.finditer(body_text))
    if not matches or len(body_text) <= MAX_GITHUB_BODY_CHARS:
        return body_text, [], None
    match = matches[-1]
    try:
        raw_payload = base64.urlsafe_b64decode(match.group("payload").encode("ascii"))
        parsed = json.loads(raw_payload.decode("utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError("recommendation object required")
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AgentLoopError(
            "Execution recommendation marker is not a recoverable JSON object."
        ) from exc
    if "$round_transport_execution_recommendation" in parsed:
        return body_text, [], None

    canonical_raw = json.dumps(
        parsed, separators=(",", ":"), sort_keys=True, ensure_ascii=False
    ).encode("utf-8")
    packed = zlib.compress(canonical_raw, 9)
    if len(packed) > _MAX_COMPRESSED:
        raise AgentLoopError("Execution recommendation is too large to transport safely.")
    encoded_packed = _b64(packed)
    anchor_id = hashlib.sha256(body_text.encode("utf-8")).hexdigest()[:24]
    spill_digest = hashlib.sha256(packed).hexdigest()
    raw_digest = hashlib.sha256(canonical_raw).hexdigest()
    chunks = [
        encoded_packed[index : index + _PART_CHARS]
        for index in range(0, len(encoded_packed), _PART_CHARS)
    ]
    reference = {
        "$round_transport_execution_recommendation": anchor_id,
        "field": "execution_recommendation",
        "parts": len(chunks),
        "sha256": raw_digest,
        "spill": spill_digest,
    }
    replacement = _b64(
        json.dumps(reference, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
    transformed = (
        body_text[: match.start("payload")]
        + replacement
        + body_text[match.end("payload") :]
    )
    sidecars = [
        _sidecar(
            {
                "v": 1,
                "anchor": anchor_id,
                "spill": spill_digest,
                "field": "execution_recommendation",
                "index": index,
                "count": len(chunks),
                "sha256": raw_digest,
                "data": chunk,
            }
        )
        for index, chunk in enumerate(chunks)
    ]

    # The complete recommendation is needed by agents and resume, but its
    # human-readable projection is deliberately unbounded.  Once the marker
    # has been moved into transport sidecars, replace the whole rendered
    # section with a bounded, explicit summary.  The canonical plan in round
    # metadata remains lossless, and the marker below lets readers hydrate the
    # same structured object from the sidecars.
    boundaries = [
        boundary
        for boundary in _EXECUTION_RECOMMENDATION_SECTION_BOUNDARY_RE.finditer(body_text)
        if (
            boundary.end() <= match.start()
            and boundary.group("digest")
            == hashlib.sha256(match.group("payload").encode("ascii")).hexdigest()
            and body_text[boundary.end() :].lstrip("\r\n").startswith(
                "### Execution strategy recommendation (v1)"
            )
        )
    ]
    if not boundaries:
        return transformed, sidecars, None
    section_boundary = boundaries[-1]
    transported_matches = list(_EXECUTION_RECOMMENDATION_RE.finditer(transformed))
    if not transported_matches:
        raise AgentLoopError(
            "Execution recommendation transport rewrite lost its protocol marker."
        )
    transported_match = transported_matches[-1]
    preserved_markers = [
        occurrence.text
        for occurrence in scan_reserved_markers(body_text)
        if (
            section_boundary.start() <= occurrence.start < match.end()
            and occurrence.definition.token != "AGENT_EXECUTION_RECOMMENDATION"
        )
    ]
    compact_lines = [
        "### Execution strategy recommendation (v1)",
    ]
    if parsed.get("strategy") in {"one-shot", "staged"}:
        compact_lines.append(f"- `strategy`: `{parsed['strategy']}`")
    if parsed.get("staging_feasibility") in {"safe", "inseparable"}:
        compact_lines.append(
            f"- `staging_feasibility`: `{parsed['staging_feasibility']}`"
        )
    compact_lines.extend(
        [
            "The complete validated execution recommendation is retained in the "
            "canonical plan metadata and bounded transport sidecars for reviewer "
            "prompts and lossless resume.",
            *preserved_markers,
            transported_match.group(0),
        ]
    )
    compact_section = "\n".join(compact_lines)
    compacted = (
        transformed[: section_boundary.start()]
        + compact_section
        + transformed[transported_match.end() :]
    )
    # The range is expressed in the original carrier so the caller can retain
    # authorization for every marker outside the rewritten recommendation.
    return compacted, sidecars, (
        section_boundary.start(),
        match.end(),
        compact_section,
    )


def _prepare_risk_test_matrix_transport(
    body_text: str,
) -> tuple[str, list[TrustedBody], tuple[int, int, str] | None]:
    """Move a large matrix marker to sidecars and retain a compact public index."""
    matches = list(_RISK_TEST_MATRIX_RE.finditer(body_text))
    if (
        not matches
        or len(body_text) <= _RISK_MATRIX_COMPACT_AT_CHARS
        or len(matches[-1].group("payload")) <= _RISK_MATRIX_MARKER_COMPACT_AT_CHARS
    ):
        return body_text, [], None
    match = matches[-1]
    try:
        raw_payload = base64.urlsafe_b64decode(match.group("payload").encode("ascii"))
        parsed = json.loads(raw_payload.decode("utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError("matrix object required")
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AgentLoopError("Risk matrix marker is not a recoverable JSON object.") from exc
    if "$round_transport_risk_test_matrix" in parsed:
        return body_text, [], None

    identity = parsed.get("identity")
    matrix = parsed.get("matrix")
    if not isinstance(identity, str) or not re.fullmatch(r"[0-9a-f]{64}", identity):
        raise AgentLoopError("Risk matrix marker has no valid identity.")
    if not isinstance(matrix, dict):
        raise AgentLoopError("Risk matrix marker has no matrix object.")

    canonical_raw = json.dumps(
        parsed, separators=(",", ":"), sort_keys=True, ensure_ascii=False
    ).encode("utf-8")
    packed = zlib.compress(canonical_raw, 9)
    if len(packed) > _MAX_COMPRESSED:
        raise AgentLoopError("Risk matrix is too large to transport safely.")
    encoded_packed = _b64(packed)
    anchor_id = hashlib.sha256(body_text.encode("utf-8")).hexdigest()[:24]
    spill_digest = hashlib.sha256(packed).hexdigest()
    raw_digest = hashlib.sha256(canonical_raw).hexdigest()
    chunks = [
        encoded_packed[index : index + _PART_CHARS]
        for index in range(0, len(encoded_packed), _PART_CHARS)
    ]
    reference = {
        "$round_transport_risk_test_matrix": anchor_id,
        "field": "risk_test_matrix_marker",
        "parts": len(chunks),
        "sha256": raw_digest,
        "spill": spill_digest,
    }
    replacement = _b64(
        json.dumps(reference, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
    transformed = (
        body_text[: match.start("payload")]
        + replacement
        + body_text[match.end("payload") :]
    )
    sidecars = [
        _sidecar(
            {
                "v": 1,
                "anchor": anchor_id,
                "spill": spill_digest,
                "field": "risk_test_matrix_marker",
                "index": index,
                "count": len(chunks),
                "sha256": raw_digest,
                "data": chunk,
            }
        )
        for index, chunk in enumerate(chunks)
    ]

    boundaries = [
        boundary
        for boundary in _RISK_TEST_MATRIX_SECTION_BOUNDARY_RE.finditer(body_text)
        if boundary.end() <= match.start() and boundary.group("identity") == identity
    ]
    if not boundaries:
        return transformed, sidecars, None
    section_boundary = boundaries[-1]
    transported_matches = list(_RISK_TEST_MATRIX_RE.finditer(transformed))
    if not transported_matches:
        raise AgentLoopError(
            "Risk test matrix transport rewrite lost its protocol marker."
        )
    transported_match = transported_matches[-1]
    rows = matrix.get("rows")
    row_index: list[tuple[str, str, str]] = []
    if isinstance(rows, list):
        for row in rows:
            row_id = row.get("row_id") if isinstance(row, dict) else None
            if isinstance(row_id, str) and re.fullmatch(r"[A-Za-z0-9._-]{1,128}", row_id):
                label = sanitize_historical_text(str(row.get("label", "")))
                owner = sanitize_historical_text(str(row.get("execution_owner", "")))
                label = " ".join(label.split()).replace("|", "\\|")
                if len(label) > 160:
                    label = label[:157].rstrip() + "..."
                row_index.append((row_id, label, owner))
    applicability = matrix.get("applicability")
    preserved_markers = [
        occurrence.text
        for occurrence in scan_reserved_markers(body_text)
        if (
            section_boundary.start() <= occurrence.start < match.end()
            and occurrence.definition.token != "AGENT_RISK_TEST_MATRIX"
        )
    ]
    compact_lines = [
        risk_test_matrix_section_boundary(identity),
        "### Risk-based mode and transition test matrix",
        f"- **Applicability:** {applicability if applicability in {'applicable', 'not-applicable'} else 'unknown'}",
        f"- **Rows:** {len(rows) if isinstance(rows, list) else 0}",
    ]
    if row_index:
        compact_lines.extend(
            [
                "",
                "| ID | Scenario | Owner |",
                "| --- | --- | --- |",
                *[
                    f"| `{row_id}` | {label} | `{owner}` |"
                    for row_id, label, owner in row_index
                ],
            ]
        )
    compact_lines.extend(
        [
            "The complete validated matrix is retained in authenticated transport "
            "sidecars and is hydrated losslessly for reviewer and coder prompts.",
            *preserved_markers,
            transported_match.group(0),
        ]
    )
    compact_section = "\n".join(compact_lines)
    compacted = (
        transformed[: section_boundary.start()]
        + compact_section
        + transformed[transported_match.end() :]
    )
    return compacted, sidecars, (
        section_boundary.start(),
        match.end(),
        compact_section,
    )


def _replace_authorized_range(
    carrier: TrustedBody,
    *,
    start: int,
    end: int,
    replacement: TrustedBody,
    replaceable_token: str,
) -> TrustedBody:
    """Replace visible text while retaining marker provenance around it."""
    original = str(carrier)
    if start < 0 or end < start or end > len(original):
        raise AgentLoopError("Cannot transport an invalid authorized text range.")

    segments = []
    cursor = 0
    inserted = False
    replacement_markers = {
        (segment.token, segment.text)
        for segment in replacement._segments
        if segment.token is not None
    }
    for segment in carrier._segments:
        segment_start = cursor
        segment_end = cursor + len(segment.text)
        overlaps = segment_start < end and segment_end > start
        if not overlaps:
            if not inserted and segment_start >= end:
                segments.extend(replacement._segments)
                inserted = True
            segments.append(segment)
        else:
            if segment.token is not None:
                if not (
                    start <= segment_start
                    and segment_end <= end
                    and (
                        segment.token == replaceable_token
                        or (segment.token, segment.text) in replacement_markers
                    )
                ):
                    raise AgentLoopError(
                        "Cannot transport a range containing an unrelated authorized marker."
                    )
            else:
                if segment_start < start:
                    segments.append(
                        type(segment)(segment.text[: start - segment_start])
                    )
                if not inserted:
                    segments.extend(replacement._segments)
                    inserted = True
                if segment_end > end:
                    segments.append(type(segment)(segment.text[end - segment_start :]))
        cursor = segment_end

    if not inserted:
        segments.extend(replacement._segments)
    return TrustedBody(
        "".join(segment.text for segment in segments),
        tuple(segment for segment in segments if segment.text),
    )


def _replace_authorized_marker(
    carrier: TrustedBody,
    *,
    token: str,
    old_text: str,
    new_text: str,
) -> TrustedBody:
    """Replace one marker while retaining the carrier's segment provenance."""
    if old_text == new_text:
        return carrier
    replaced = False
    segments = []
    for segment in carrier._segments:
        if not replaced and segment.token == token and segment.text == old_text:
            segments.append(type(segment)(new_text, token))
            replaced = True
        else:
            segments.append(segment)
    if not replaced:
        raise AgentLoopError(
            f"Cannot transport {token}: its authorized marker segment was not found."
        )
    return TrustedBody(
        "".join(segment.text for segment in segments), tuple(segments)
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
    body_text, sidecars, execution_rewrite = _prepare_execution_recommendation_transport(
        body_text
    )
    trusted_anchor = carrier
    if execution_rewrite is not None:
        start, end, compact_section = execution_rewrite
        trusted_compact_section = TrustedBody.canonical(
            compact_section,
            expected_tokens=tuple(
                occurrence.definition.token
                for occurrence in scan_reserved_markers(compact_section)
            ),
        )
        trusted_anchor = _replace_authorized_range(
            trusted_anchor,
            start=start,
            end=end,
            replacement=trusted_compact_section,
            replaceable_token="AGENT_EXECUTION_RECOMMENDATION",
        )
    elif sidecars:
        original_execution = _EXECUTION_RECOMMENDATION_RE.search(str(carrier))
        transported_execution = _EXECUTION_RECOMMENDATION_RE.search(body_text)
        if original_execution is not None and transported_execution is not None:
            trusted_anchor = _replace_authorized_marker(
                trusted_anchor,
                token="AGENT_EXECUTION_RECOMMENDATION",
                old_text=original_execution.group(0),
                new_text=transported_execution.group(0),
            )
    body_text, matrix_sidecars, matrix_rewrite = _prepare_risk_test_matrix_transport(
        body_text
    )
    sidecars.extend(matrix_sidecars)
    if matrix_rewrite is not None:
        start, end, compact_section = matrix_rewrite
        trusted_compact_section = TrustedBody.canonical(
            compact_section,
            expected_tokens=tuple(
                occurrence.definition.token
                for occurrence in scan_reserved_markers(compact_section)
            ),
        )
        trusted_anchor = _replace_authorized_range(
            trusted_anchor,
            start=start,
            end=end,
            replacement=trusted_compact_section,
            replaceable_token="AGENT_RISK_TEST_MATRIX",
        )
    elif matrix_sidecars:
        original_matrices = list(_RISK_TEST_MATRIX_RE.finditer(str(trusted_anchor)))
        transported_matrices = list(_RISK_TEST_MATRIX_RE.finditer(body_text))
        if original_matrices and transported_matrices:
            trusted_anchor = _replace_authorized_marker(
                trusted_anchor,
                token="AGENT_RISK_TEST_MATRIX",
                old_text=original_matrices[-1].group(0),
                new_text=transported_matrices[-1].group(0),
            )
    matches = list(ROUND_RESUME_MARKER_RE.finditer(body_text))
    if len(body_text) > MAX_GITHUB_BODY_CHARS and not matches and not sidecars:
        raise AgentLoopError(
            f"GitHub comment body exceeds {MAX_GITHUB_BODY_CHARS} characters; shorten the response."
        )
    if not matches:
        if sidecars:
            if len(body_text) > MAX_GITHUB_BODY_CHARS:
                raise AgentLoopError(
                    f"GitHub comment body exceeds {MAX_GITHUB_BODY_CHARS} characters after execution sidecar spill."
                )
            return (*sidecars, trusted_anchor)
        # Preserve the caller's authorization when no transport rewrite was
        # needed. Re-scanning this same text would authorize markers that the
        # caller did not authorize at composition time.
        return (carrier,)

    # Resume reads the last marker when a legacy comment contains more than one.
    match = matches[-1]
    payload = decode_mapping(match.group("payload"))
    sidecars = list(sidecars)
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
    original_round = ROUND_RESUME_MARKER_RE.search(str(trusted_anchor))
    transported_round = ROUND_RESUME_MARKER_RE.search(anchor)
    if original_round is None or transported_round is None:
        raise AgentLoopError("Cannot transport AGENT_LOOP_META without its authorized marker segment.")
    trusted_anchor = _replace_authorized_marker(
        trusted_anchor,
        token="AGENT_LOOP_META",
        old_text=original_round.group(0),
        new_text=transported_round.group(0),
    )
    return (*sidecars, trusted_anchor)


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
        if not isinstance(ref, dict):
            continue
        reference_key = (
            "$round_transport_execution_recommendation"
            if "$round_transport_execution_recommendation" in ref
            else (
                "$round_transport_risk_test_matrix"
                if "$round_transport_risk_test_matrix" in ref
                else "$round_transport_spill"
            )
        )
        if reference_key not in ref:
            continue
        try:
            anchor = str(ref[reference_key])
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
