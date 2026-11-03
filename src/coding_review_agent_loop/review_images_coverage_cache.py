"""Private sealed-byte availability cache; authenticated attachments are authority."""
from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile

from .review_images_coverage import fail
from .errors import AgentLoopError
from .review_images_config import delivery_limits
from .review_images_state import digest, sha
from .scratch import scratch_root, make_private_dirs


def root():
    path = make_private_dirs(scratch_root() / 'coverage-evidence')
    return path


def directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    info = os.fstat(fd)
    if info.st_uid != os.getuid() or info.st_mode & 0o077:
        os.close(fd)
        fail('cache directory is not private')
    return fd


def write_file(fd, name, data):
    target = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=fd)
    try:
        with os.fdopen(target, 'wb', closefd=False) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(target)
        os.fchmod(target, 0o400)
    finally:
        os.close(target)


def read_file(fd, name, cap):
    target = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
    try:
        info = os.fstat(target)
        if (not stat.S_ISREG(info.st_mode) or info.st_size > cap or info.st_size < 1
                or info.st_uid != os.getuid() or info.st_mode & 0o077 or info.st_nlink != 1):
            fail('invalid cache artifact')
        data = os.read(target, cap + 1)
        if len(data) != info.st_size:
            fail('cache artifact changed')
        return data
    finally:
        os.close(target)


def persist(snapshots, binding, *, proof, source_id=None):
    limits = delivery_limits()
    manifest = json.loads(snapshots.manifest_json)
    attachment = {'version': 1, 'head': snapshots.head_sha, 'assessment_id': binding['attempt'],
        'snapshot_digest': snapshots.digest, 'configuration': snapshots.config_digest,
        'proof': proof, 'source_id': source_id,
        'images': [{key: image[key] for key in ('id', 'viewport', 'locale', 'state')}
                   for image in manifest['images']],
        'manifest': manifest, 'inventory': []}
    parent = root()
    parent_fd = directory(parent)
    try:
        name = os.path.basename(tempfile.mkdtemp(prefix='evidence-', dir=parent))
        fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
        try:
            total = 0
            for i, file in enumerate(snapshots.files):
                data = file.read()
                total += len(data)
                if total > limits.total_bytes or len(data) > limits.file_bytes:
                    fail('cache exceeds image budget')
                if hashlib.sha256(data).hexdigest() != file.digest:
                    fail('sealed bytes changed')
                write_file(fd, str(i), data)
                attachment['inventory'].append({'name': str(i), 'path': file.path, 'digest': file.digest,
                    'size': len(data), 'width': file.width, 'height': file.height})
            write_file(fd, 'inventory', json.dumps(attachment, sort_keys=True).encode())
        finally:
            os.close(fd)
        attachment['cache'] = name
        attachment['inventory_digest'] = digest({k: v for k, v in attachment.items() if k != 'cache'})
        return attachment
    finally:
        os.close(parent_fd)


def available(attachment):
    """Recompute snapshot binding from bounded regular files on every consumption."""
    if not isinstance(attachment, dict) or set(attachment) != {'version', 'head', 'assessment_id',
            'snapshot_digest', 'configuration', 'proof', 'source_id', 'images', 'manifest',
            'inventory', 'cache', 'inventory_digest'}:
        return False
    name = attachment['cache']
    if not isinstance(name, str) or not name.startswith('evidence-') or '/' in name or '..' in name:
        return False
    parent_fd = fd = None
    try:
        parent_fd = directory(root())
        fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
        info = os.fstat(fd)
        if info.st_uid != os.getuid() or info.st_mode & 0o077:
            return False
        stored = json.loads(read_file(fd, 'inventory', 1024 * 1024))
        expected = {k: v for k, v in attachment.items() if k not in {'cache', 'inventory_digest'}}
        if stored != expected or digest(stored) != attachment['inventory_digest']:
            return False
        limits = delivery_limits()
        expected_images = [{key: image[key] for key in ('id', 'viewport', 'locale', 'state')}
                           for image in stored['manifest']['images']]
        if stored['images'] != expected_images:
            return False
        inventory = stored['inventory']
        if not isinstance(inventory, list) or not 0 < len(inventory) <= limits.entries * 2:
            return False
        total = 0
        files = []
        for i, entry in enumerate(inventory):
            if entry['name'] != str(i):
                return False
            data = read_file(fd, str(i), limits.file_bytes)
            total += len(data)
            if total > limits.total_bytes or len(data) != entry['size'] or hashlib.sha256(data).hexdigest() != entry['digest']:
                return False
            files.append((entry['path'], entry['digest']))
        computed = digest({'manifest': stored['manifest'], 'config': stored['configuration'], 'files': files})
        return (computed == stored['snapshot_digest'] and stored['manifest']['head_sha'] == stored['head']
                and sha(stored['proof'], 64))
    except (OSError, ValueError, TypeError, KeyError, AgentLoopError):
        return False
    finally:
        if fd is not None:
            os.close(fd)
        if parent_fd is not None:
            os.close(parent_fd)
