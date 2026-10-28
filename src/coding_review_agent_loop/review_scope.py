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
    return get("role") == "reviewer" and not scope_present(metadata)


def encode_review_scope(metadata):
    return {**({"review_scope": metadata.review_scope} if scope_present(metadata) else {}),
            **({"capture_review_binding": metadata.capture_review_binding}
               if metadata.capture_review_binding is not None else {})}


def decode_review_scope(payload):
    return {**({"review_scope": payload["review_scope"], "review_scope_present": True}
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
