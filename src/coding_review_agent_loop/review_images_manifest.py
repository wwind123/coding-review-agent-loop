"""Independent parent manifest validation and immutable caller-owned snapshots."""
from __future__ import annotations
from dataclasses import dataclass, field
import fcntl
import hashlib
import json
import math
import unicodedata
import os

from .review_images_channel import loads
from .review_images_config import ImageLimits, relative_path


def text(value, limit):
    if not isinstance(value, str) or not value or len(value) > limit or any(unicodedata.category(c).startswith('C') for c in value):
        raise ValueError('malformed manifest text')
    return value


def schema(raw: bytes, head: str, limits: ImageLimits):
    if len(raw) > limits.manifest_bytes:
        raise ValueError('manifest exceeds byte limit')
    value = loads(raw)
    if not isinstance(value, dict) or set(value) != {'head_sha', 'images'} or value['head_sha'] != head:
        raise ValueError('manifest head/schema mismatch')
    images = value['images']
    if not isinstance(images, list) or not 0 < len(images) <= limits.entries:
        raise ValueError('manifest image count out of bounds')
    ids = set()
    paths = []
    for image in images:
        required = {'path', 'id', 'viewport', 'locale', 'state'}
        if (not isinstance(image, dict) or not required <= set(image)
                or set(image) - required - {'baseline_path', 'diff_score'}):
            raise ValueError('invalid image properties')
        identity = text(image['id'], 128)
        if identity in ids:
            raise ValueError('duplicate image id')
        ids.add(identity)
        text(image['locale'], 64)
        text(image['state'], 256)
        viewport = image['viewport']
        if not isinstance(viewport, dict) or set(viewport) != {'width', 'height'}:
            raise ValueError('invalid viewport')
        if any(isinstance(n, bool) or not isinstance(n, int) or not 0 < n <= limits.viewport_dimension for n in viewport.values()):
            raise ValueError('invalid viewport dimensions')
        if 'diff_score' in image:
            score = image['diff_score']
            try:
                finite = math.isfinite(score)
            except (TypeError, OverflowError):
                finite = False
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not finite:
                raise ValueError('invalid advisory diff score')
        for name in ('path', 'baseline_path'):
            if name in image:
                paths.append(relative_path(image[name]))
    return value, paths


@dataclass(frozen=True)
class SnapshotFile:
    path: str
    digest: str
    width: int
    height: int
    size: int
    fd: int
    _closed: bool = field(default=False, init=False, repr=False, compare=False)

    def read(self):
        if self._closed:
            raise OSError('snapshot is closed')
        return os.pread(self.fd, self.size, 0)


@dataclass(frozen=True)
class SnapshotSet:
    head_sha: str
    manifest_json: bytes
    config_digest: str
    digest: str
    files: tuple[SnapshotFile, ...]

    def close(self):
        for file in self.files:
            if not file._closed:
                object.__setattr__(file, '_closed', True)
                try:
                    os.close(file.fd)
                except OSError:
                    pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def publish(raw, head, limits, content, dimensions, config_digest):
    value, paths = schema(raw, head, limits)
    if not isinstance(content, dict) or not isinstance(dimensions, dict):
        raise ValueError('malformed validator reply')
    if set(paths) != set(content) or set(dimensions) != set(content):
        raise ValueError('export inventory mismatch')
    total_bytes = total_pixels = 0
    for path in paths:
        width, height = dimensions[path]
        if any(isinstance(n, bool) or not isinstance(n, int) or not 0 < n <= limits.dimension for n in (width, height)):
            raise ValueError('invalid validator dimensions')
        data = content[path]
        total_bytes += len(data)
        total_pixels += width * height
        if len(data) > limits.file_bytes or width * height > limits.pixels:
            raise ValueError('image exceeds limit')
    if total_bytes > limits.total_bytes or total_pixels > limits.total_pixels:
        raise ValueError('aggregate image limit exceeded')
    files = []
    try:
        for path in sorted(content):
            data = content[path]
            fd = os.memfd_create('review-image', os.MFD_ALLOW_SEALING | os.MFD_CLOEXEC)
            try:
                with os.fdopen(os.dup(fd), 'wb') as stream:
                    stream.write(data)
                fcntl.fcntl(fd, fcntl.F_ADD_SEALS, fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL)
                readonly = os.open(f'/proc/self/fd/{fd}', os.O_RDONLY | os.O_CLOEXEC)
            finally:
                os.close(fd)
            files.append(SnapshotFile(path, hashlib.sha256(data).hexdigest(), *dimensions[path], len(data), readonly))
        canonical = json.dumps({'manifest': value, 'config': config_digest,
                                'files': [(f.path, f.digest) for f in files]}, sort_keys=True, separators=(',', ':')).encode()
        return SnapshotSet(head, json.dumps(value, sort_keys=True).encode(), config_digest,
                           hashlib.sha256(canonical).hexdigest(), tuple(files))
    except BaseException:
        for file in files:
            os.close(file.fd)
        raise
