"""Stage-A transition shim for the callee-owned shard plugin (#1210).

The plugin now lives in ``ci/managed/ci_shard_plugin.py`` and is injected with
``-p ci_shard_plugin`` by the reusable workflow.  Until the default-branch
workflow *is* that reusable workflow, the installed (old) workflow still runs a
plain ``python -m pytest -n auto`` with ``CI_SHARD_*`` set, so this repository's
conftest keeps loading this module.  It registers the real plugin only when it
is not already registered, so injection plus this shim never registers it twice.

Removed together with ``tests/ci_shard_verify.py`` once the default branch uses
the reusable workflow.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_CALLEE = str(_ROOT / "ci" / "managed")
if _CALLEE not in sys.path:
    # Appended: the tests directory (and its verifier shim) must keep precedence.
    sys.path.append(_CALLEE)

import ci_shard_plugin as _plugin  # noqa: E402

# This repository's committed balance data; ``CI_SHARD_DURATIONS`` still wins.
_plugin.DEFAULT_DURATIONS = Path(__file__).parent / ".test_durations"

# Re-exported helpers only.  No ``pytest_*`` hook or fixture may be re-exported:
# that would register it a second time on this module.
SHARD_ENV_VARS = _plugin.SHARD_ENV_VARS
ShardConfig = _plugin.ShardConfig
parse_env = _plugin.parse_env
partition = _plugin.partition
check_exactly_once = _plugin.check_exactly_once
_CONFIG_KEY = _plugin._CONFIG_KEY
_FULL_KEY = _plugin._FULL_KEY
_DURATIONS_KEY = _plugin._DURATIONS_KEY
_load_durations = _plugin._load_durations
_owns_manifest = _plugin._owns_manifest
_owns_durations = _plugin._owns_durations


def pytest_configure(config):
    manager = config.pluginmanager
    if not manager.has_plugin("ci_shard_plugin") and not manager.is_registered(_plugin):
        manager.register(_plugin, "ci_shard_plugin")
