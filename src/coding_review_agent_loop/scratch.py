"""Private creation of agent-loop's shared scratch tree.

Everything agent-loop keeps under ``$TMPDIR/coding-review-agent-loop`` shares
one top-level directory, and sandboxed mode refuses a response root whose
components are group- or world-writable.  ``Path.mkdir(parents=True)`` gives
every missing ancestor the process umask (``775`` under the common
``umask 002``), so one non-sandboxed run could leave a tree that a later
sandboxed run then refuses.  :func:`make_private_dirs` creates each missing
component with an explicit ``0o700`` instead.  Existing directories are left
untouched: the sandbox validation remains the backstop for a pre-existing or
foreign tree.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

SCRATCH_DIR_NAME = "coding-review-agent-loop"
PRIVATE_DIR_MODE = 0o700


def scratch_root() -> Path:
    """The lexical top-level scratch directory under the current TMPDIR."""
    return Path(tempfile.gettempdir()) / SCRATCH_DIR_NAME


def make_private_dirs(path: Path | str) -> Path:
    """Create ``path`` and every missing ancestor with mode ``0o700``.

    ``os.mkdir``'s mode argument is masked by the umask, which can only clear
    bits, so the created directories are never group- or world-writable
    whatever the caller's umask.  Components that already exist keep their
    mode.
    """
    target = Path(os.path.abspath(path))
    missing: list[Path] = []
    current = target
    while not os.path.lexists(current):
        missing.append(current)
        parent = current.parent
        if parent == current:
            break
        current = parent
    for directory in reversed(missing):
        try:
            os.mkdir(directory, PRIVATE_DIR_MODE)
        except FileExistsError:
            pass
    if not target.is_dir():
        raise NotADirectoryError(f"Not a directory: {target}")
    return target
