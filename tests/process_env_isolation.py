"""Restore the process environment around each test.

Code under test can mutate ``os.environ`` directly. For example,
``agent_permissions.establish_sandboxed_run`` applies ``neutralized_git_env()``
with ``os.environ.update``, and ``monkeypatch`` never sees that write. Without a
restore, ``GIT_EXTERNAL_DIFF=""`` survives into every later test on the same
xdist worker, and each ``git diff`` there fails with "external diff died".
Which test meets the leak depends on shard composition (#1990 CI).
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
