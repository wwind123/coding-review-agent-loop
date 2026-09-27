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


def mkdir_private(path: Path | str) -> bool:
    """Create one directory with mode exactly ``0o700``; False if it exists.

    ``os.mkdir``'s mode is masked by the umask, which can clear owner bits
    too (``umask 0o700`` would yield mode ``000`` and an unusable
    directory), so a newly created directory is explicitly re-moded through
    a no-follow descriptor.  A component swapped for a symlink after
    creation is never re-moded through the link: re-opening it fails and
    the error propagates (fail closed).
    """
    try:
        os.mkdir(path, PRIVATE_DIR_MODE)
    except FileExistsError:
        return False
    _restore_private_mode(path)
    return True


def _restore_private_mode(path: Path | str) -> None:
    if not hasattr(os, "fchmod"):
        # Windows: POSIX mode bits do not apply.
        return
    nofollow = os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        fd = os.open(path, os.O_RDONLY | nofollow)
    except PermissionError:
        # The umask removed the owner's read bit.  An O_PATH descriptor
        # needs no permission on the directory itself, and chmod through its
        # /proc/self/fd link acts on that exact inode, never a later
        # replacement at ``path``.
        o_path = getattr(os, "O_PATH", None)
        if o_path is None or not os.path.isdir("/proc/self/fd"):
            raise PermissionError(
                f"Created {path}, but the umask removed owner access and the directory "
                "cannot be safely re-opened to restore mode 700. Use a umask that keeps "
                "owner permissions, for example 077."
            ) from None
        fd = os.open(path, o_path | nofollow)
        try:
            os.chmod(f"/proc/self/fd/{fd}", PRIVATE_DIR_MODE)
        finally:
            os.close(fd)
        return
    try:
        os.fchmod(fd, PRIVATE_DIR_MODE)
    finally:
        os.close(fd)


def make_private_dirs(path: Path | str) -> Path:
    """Create ``path`` and every missing ancestor with mode ``0o700``.

    Created directories get exactly ``0o700`` whatever the caller's umask
    (see :func:`mkdir_private`).  Components that already exist keep their
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
        mkdir_private(directory)
    if not target.is_dir():
        raise NotADirectoryError(f"Not a directory: {target}")
    return target
