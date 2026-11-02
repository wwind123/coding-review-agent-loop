"""Operator-owned, library-only capture configuration and pure eligibility."""
from __future__ import annotations

from dataclasses import dataclass, field
from fnmatch import fnmatchcase
import math
import unicodedata
from pathlib import PurePosixPath


def relative_path(value: object) -> str:
    if (not isinstance(value, str) or not value or len(value) > 1024
            or any(unicodedata.category(c).startswith('C') for c in value)
            or '\\' in value or ':' in value or value.startswith('/')
            or any(p in {'', '.', '..', '.git'} for p in value.split('/'))):
        raise ValueError('unsafe repository-relative path')
    return value


def positive(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be positive and finite')


@dataclass(frozen=True)
class ImageLimits:
    entries: int = 16
    manifest_bytes: int = 256 * 1024
    file_bytes: int = 8 * 1024**2
    total_bytes: int = 64 * 1024**2
    pixels: int = 16_000_000
    total_pixels: int = 64_000_000
    dimension: int = 16_000
    # Browser viewport metadata describes the original view, before downscaling.
    viewport_dimension: int = 16_000

    def __post_init__(self):
        for name, value in vars(self).items():
            positive(value, name)
            if not isinstance(value, int):
                raise ValueError(f'{name} must be an integer')
        if self.viewport_dimension > 16_000:
            raise ValueError('viewport metadata dimensions must not exceed 16000')


def delivery_limits(limits: ImageLimits | None = None) -> ImageLimits:
    """One effective policy for capture and transport, without silent clamping.

    Capture's standalone library defaults remain independent. Integrated callers
    pass this result to CaptureConfig and acquire_images so lower operator bounds
    apply to the same set, including baselines and aggregate reference counts.
    """
    maximum = ImageLimits(file_bytes=2 * 1024**2, pixels=768 * 768, dimension=768)
    if limits is None:
        return maximum
    if not isinstance(limits, ImageLimits):
        raise ValueError('delivery limits must be ImageLimits')
    for name, value in vars(limits).items():
        if value > getattr(maximum, name):
            raise ValueError(f'{name} exceeds admitted image transport bound')
    return limits


@dataclass(frozen=True)
class CaptureConfig:
    paths: tuple[str, ...] = ()
    command: str | None = None
    setup: str | None = None
    runtime_grants: tuple[str, ...] = ()
    setup_timeout: float = 300
    capture_timeout: float = 300
    transfer_timeout: float = 30
    validation_timeout: float = 30
    scratch_bytes: int = 2 * 1024**3
    memory_bytes: int = 1024**3
    tasks: int = 128
    cpu_percent: int = 100
    limits: ImageLimits = field(default_factory=ImageLimits)

    def __post_init__(self):
        for name in ('setup_timeout', 'capture_timeout', 'transfer_timeout', 'validation_timeout',
                     'scratch_bytes', 'memory_bytes', 'tasks', 'cpu_percent'):
            positive(getattr(self, name), name)
        for name in ('scratch_bytes', 'memory_bytes', 'tasks', 'cpu_percent'):
            if not isinstance(getattr(self, name), int):
                raise ValueError(f'{name} must be an integer')
        if not isinstance(self.paths, tuple) or not isinstance(self.runtime_grants, tuple):
            raise ValueError('paths and runtime grants must be tuples')
        for pattern in self.paths:
            relative_path(pattern)
        for command in (self.command, self.setup):
            if command is not None and (not isinstance(command, str) or not command.strip() or '\x00' in command):
                raise ValueError('invalid operator command')
        if self.paths and self.command is None:
            raise ValueError('enabled capture requires an operator command')
        for grant in self.runtime_grants:
            if not isinstance(grant, str) or not PurePosixPath(grant).is_absolute():
                raise ValueError('runtime grant must be absolute')


def eligible(config: CaptureConfig, changes: list[dict[str, str]]) -> bool:
    """Consume normalized GitHub-style path records; perform no filesystem I/O."""
    names = []
    for record in changes:
        if not isinstance(record, dict) or set(record) - {'status', 'filename', 'previous_filename'}:
            raise ValueError('malformed changed-path record')
        status = record.get('status')
        if not isinstance(status, str):
            raise ValueError('unsupported changed-path status')
        status = {'copied': 'added', 'changed': 'modified'}.get(status, status)
        if status not in {'added', 'modified', 'renamed', 'removed', 'deleted'}:
            raise ValueError('unsupported changed-path status')
        names.append(_git_path(record.get('filename')))
        if status == 'renamed':
            names.append(_git_path(record.get('previous_filename')))
        elif 'previous_filename' in record:
            raise ValueError('old name is only valid for renames')
    matched = [name for name in names if any(fnmatchcase(name, pattern) for pattern in config.paths)]
    # Git permits literal colon/backslash/control characters. They must not
    # poison unrelated nonmatches, but the capture archive cannot represent
    # matching unsupported names safely. Refuse those before reservation.
    for name in matched:
        relative_path(name)
    return bool(matched)


def _git_path(value: object) -> str:
    if (not isinstance(value, str) or not value or len(value) > 1024
            or '\x00' in value or value.startswith('/')
            or any(part in {'', '.', '..', '.git'} for part in value.split('/'))):
        raise ValueError('unsafe Git path')
    return value
