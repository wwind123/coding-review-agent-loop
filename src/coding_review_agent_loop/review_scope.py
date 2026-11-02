"""Pure classification of reviewer authority, shared by decoded and raw readers.

Presence is significant: an explicit null or unknown scope never acquires the
legacy complete-review authority. Keep the original JSON through round trips.
"""
from collections.abc import Mapping


def scope_present(metadata):
    if isinstance(metadata, Mapping):
        return "review_scope" in metadata
    return getattr(metadata, "review_scope_present", False) or getattr(metadata, "review_scope", None) is not None


def is_complete_code_review(metadata):
    get = metadata.get if isinstance(metadata, Mapping) else lambda key: getattr(metadata, key, None)
    return get("role") == "reviewer" and not scope_present(metadata) and not visual_candidate(metadata)


def encode_review_scope(metadata):
    return {**({"visual_lifecycle": metadata.visual_lifecycle} if lifecycle_present(metadata) else {}),
            **({"review_scope": metadata.review_scope} if scope_present(metadata) else {}),
            **({"capture_review_binding": metadata.capture_review_binding}
               if metadata.capture_review_binding is not None else {})}


def decode_review_scope(payload):
    return {**({"visual_lifecycle": payload["visual_lifecycle"], "visual_lifecycle_present": True}
               if "visual_lifecycle" in payload else {}),
            **({"review_scope": payload["review_scope"], "review_scope_present": True}
               if "review_scope" in payload else {}),
            **({"capture_review_binding": payload["capture_review_binding"]}
               if "capture_review_binding" in payload else {})}


def valid_evidence_scope(scope):
    return (isinstance(scope, Mapping)
            and set(scope) == {"version", "kind", "capture", "head", "matrix", "rows", "contract"}
            and type(scope["version"]) is int and scope["version"] == 1 and scope["kind"] == "capture-evidence"
            and all(isinstance(scope[key], str) and scope[key]
                    for key in ("capture", "head", "matrix", "contract"))
            and isinstance(scope["rows"], list) and bool(scope["rows"])
            and all(isinstance(row, str) and row for row in scope["rows"])
            and len(set(scope["rows"])) == len(scope["rows"]))


def is_visual_review(metadata):
    """Recognize the legacy four-key visual assessment scope.

    Use visual_candidate for quarantine and identity hydration: lifecycle
    candidates must stay outside unrelated authority even when malformed.
    """
    get = metadata.get if isinstance(metadata, Mapping) else lambda key: getattr(metadata, key, None)
    scope = get("review_scope")
    return (get("flow") == "pr" and get("role") == "reviewer" and isinstance(scope, Mapping)
            and set(scope) == {"version", "kind", "head", "seat"}
            and type(scope["version"]) is int and scope["version"] == 1
            and scope["kind"] == "visible-defects"
            and isinstance(scope["head"], str) and len(scope["head"]) == 40
            and all(c in "0123456789abcdef" for c in scope["head"])
            and scope["head"] == get("subject") and scope["seat"] == get("agent")
            and isinstance(scope["seat"], str) and bool(scope["seat"]))


def authority_records(records):
    return tuple(record for record in records if not visual_candidate(record.metadata))


def is_visual_item(item):
    return getattr(item, "origin_scope", None) == "visible-defects"


def lifecycle_present(metadata):
    if isinstance(metadata, Mapping):
        return "visual_lifecycle" in metadata
    return getattr(metadata, "visual_lifecycle_present", False) or getattr(metadata, "visual_lifecycle", None) is not None


def visual_candidate(metadata):
    """Broad quarantine before authentication or strict lifecycle validation."""
    get = metadata.get if isinstance(metadata, Mapping) else lambda key: getattr(metadata, key, None)
    scope = get("review_scope")
    return lifecycle_present(metadata) or is_visual_review(metadata) or (
        isinstance(scope, Mapping) and scope.get("kind") == "visible-defects")
