"""Stage-A transition shim: the verifier now lives in ``ci/managed/ci_shard_verify.py``.

The old installed workflow checks out ``tests/ci_shard_verify.py`` from the
*default branch*, not from this head, so this file only keeps the historical
import path and CLI working.  Stdlib only; removed with ``tests/_ci_shard.py``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_PATH = Path(__file__).resolve().parents[1] / "ci" / "managed" / "ci_shard_verify.py"
_spec = importlib.util.spec_from_file_location("_managed_ci_shard_verify_impl", _PATH)
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)

SCHEMA_VERSION = _impl.SCHEMA_VERSION
MANIFEST_FIELDS = _impl.MANIFEST_FIELDS
collection_digest = _impl.collection_digest
verify = _impl.verify
main = _impl.main

if __name__ == "__main__":
    sys.exit(main())
