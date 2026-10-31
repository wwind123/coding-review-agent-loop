"""Private exact-head Git staging; repository configuration never executes."""
from __future__ import annotations
import os
from contextlib import contextmanager
import shutil
from pathlib import Path
import re
import selectors
import time
import subprocess
import tempfile

from .secure_git import local_command
from .review_images_config import relative_path
from .review_images_channel import write_record


def git(root, args, *, input=None, output=None, cap=16*1024**2):
    argv, env, fds = local_command(['-C', str(root), *args], checkout=root)
    # Never buffer an unbounded Git response or let pack output fill host disk.
    with tempfile.TemporaryFile() as source:
        if input is not None:
            source.write(input)
            source.seek(0)
        process = subprocess.Popen(argv, env=env, pass_fds=fds, stdin=source,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ, 'out')
        selector.register(process.stderr, selectors.EVENT_READ, 'err')
        result = bytearray()
        error = bytearray()
        total = 0
        deadline = time.monotonic() + 30
        try:
            while selector.get_map():
                if time.monotonic() >= deadline:
                    raise TimeoutError('confined Git staging deadline')
                for key, _ in selector.select(.05):
                    data = os.read(key.fileobj.fileno(), 65536)
                    if not data:
                        selector.unregister(key.fileobj)
                    elif key.data == 'err':
                        error.extend(data)
                        del error[:-4096]
                    else:
                        total += len(data)
                        if total > cap:
                            raise ValueError('Git staging exceeds byte cap')
                        if output is not None:
                            output.write(data)
                        else:
                            result.extend(data)
            code = process.wait(timeout=max(.1, deadline-time.monotonic()))
            if code:
                raise ValueError('confined Git staging failed: ' + error.decode(errors='replace'))
            return bytes(result)
        finally:
            selector.close()
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            process.stdout.close()
            process.stderr.close()


@contextmanager
def owned_staging():
    directory = Path(tempfile.mkdtemp(prefix='review-images-stage-'))
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    original = os.fstat(fd)
    try:
        yield directory
    finally:
        try:
            current = directory.lstat()
            if (current.st_dev, current.st_ino) != (original.st_dev, original.st_ino):
                raise ValueError('staging root ownership changed; refusing cleanup')
            # shutil's fd-based rmtree does not follow tracked symlink targets.
            shutil.rmtree(directory)
        finally:
            os.close(fd)


def stage(checkout: Path, head: str, cap: int, destination):
    staging_deadline = time.monotonic() + 30
    if not isinstance(head, str) or not re.fullmatch(r'(?:[0-9a-f]{40}|[0-9a-f]{64})', head):
        raise ValueError('capture requires a full pinned commit OID')
    if git(checkout, ['rev-parse', '--verify', head + '^{commit}']).decode().strip() != head:
        raise ValueError('requested head is not a commit')
    # Reject expansion before importing a compressed pack into host staging.
    raw_tree = git(checkout, ['ls-tree', '-rlz', '--full-tree', head], cap=min(cap, 128*1024**2))
    entries = [record for record in raw_tree.split(b'\0') if record]
    if len(entries) > 100000:
        raise ValueError('too many tracked entries')
    sizes = []
    for entry in entries:
        attrs, path = entry.split(b'\t', 1)
        mode, kind, oid, size = attrs.decode().split()
        relative_path(path.decode('utf8'))
        if mode not in {'100644', '100755', '120000'} or kind != 'blob':
            raise ValueError('unsupported tracked layout')
        sizes.append(int(size))
    if sum(sizes) > cap or int(git(checkout, ['cat-file', '-s', head])) > cap:
        raise ValueError('expanded tracked tree exceeds staging cap')
    with owned_staging() as private:
        git(private, ['init', '--quiet', '--object-format=' + ('sha256' if len(head) == 64 else 'sha1')])
        # Pack only this commit and its tree, no traversal of history or reuse
        # via alternates/hardlinks. The private repository has no repo config.
        objects = git(checkout, ['rev-list', '--objects', '--no-object-names', head + '^{tree}'], cap=min(cap, 128*1024**2))
        with tempfile.TemporaryFile() as pack:
            git(checkout, ['pack-objects', '--stdout'], input=head.encode() + b'\n' + objects, output=pack, cap=cap)
            if pack.tell() > cap:
                raise ValueError('staging pack exceeds scratch cap')
            pack.seek(0)
            argv, env, fds = local_command(['-C', str(private), 'unpack-objects', '-q'], checkout=private)
            done = subprocess.run(argv, stdin=pack, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                  env=env, pass_fds=fds, timeout=30)
            if done.returncode:
                raise ValueError('private pack import failed')
        git(private, ['checkout', '--quiet', '--detach', head])
        git(private, ['diff', '--exit-code', head, '--'])
        if git(private, ['rev-parse', 'HEAD']).decode().strip() != head:
            raise ValueError('private worktree head mismatch')
        inventory = git(private, ['ls-tree', '-rz', '--full-tree', 'HEAD']).split(b'\0')
        total = count = 0
        seen = set()
        for item in inventory:
            if time.monotonic() >= staging_deadline:
                raise TimeoutError('whole-tree staging deadline')
            if not item:
                continue
            attrs, raw_path = item.split(b'\t', 1)
            mode, kind, oid = attrs.decode().split()
            path = relative_path(raw_path.decode('utf-8', errors='strict'))
            if path in seen or mode not in {'100644', '100755', '120000'} or kind != 'blob':
                raise ValueError('unsupported tracked layout')
            seen.add(path)
            content = git(private, ['cat-file', 'blob', oid], cap=cap)
            total += len(content)
            count += 1
            if total > cap or count > 100000:
                raise ValueError('tracked tree exceeds capture input bounds')
            write_record(destination, {'path': path, 'mode': mode, 'size': len(content)}, content)
        write_record(destination, {'complete': True})
