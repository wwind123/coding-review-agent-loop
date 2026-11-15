"""Restore the process environment around each test.

Code under test can mutate ``os.environ`` directly. For example,
``agent_permissions.establish_sandboxed_run`` applies the neutralized git
environment (overrides and removals) directly to ``os.environ``, and
``monkeypatch`` never sees that write. Without a restore, the direct write
survives into every later test on the same xdist worker and changes how git
behaves there. Which test meets the leak depends on shard composition (#1990 CI).
"""
from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def restored_process_environment() -> Iterator[None]:
    """Snapshot ``os.environ`` and restore it exactly, keys added or changed alike."""
    snapshot = dict(os.environ)
    try:
        yield
    finally:
        if dict(os.environ) != snapshot:
            os.environ.clear()
            os.environ.update(snapshot)
