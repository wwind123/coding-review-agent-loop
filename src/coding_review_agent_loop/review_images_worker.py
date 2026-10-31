"""Trusted sandbox supervisor. Installed as sealed read-only source by host."""
import ctypes
import errno
import fcntl
import json
import os
from pathlib import Path
import resource
import selectors
import socket
import stat
import subprocess
import sys
import time

sys.path.insert(0, '/trusted')
from review_images_channel import records, write_record
from review_images_config import ImageLimits, relative_path
from review_images_manifest import schema
from review_images_png import validate_png
from review_images_mounts import verify_mounts, DEVICES


def mount_probe(size):
    info = os.statvfs('/scratch')
    proof = verify_mounts(Path('/proc/self/mountinfo').read_text(),
                          info.f_blocks * info.f_frsize, size, os.sysconf('SC_PAGE_SIZE'))
    for device in DEVICES:
        if not stat.S_ISCHR(os.stat(device).st_mode):
            raise ValueError('unexpected non-device binding')
    for name in ('home', 'tmp', 'shm', 'work', 'output'):
        Path('/scratch', name).mkdir()
    probe = Path('/scratch/tmp/exec-probe')
    probe.write_text('#!/usr/bin/python3\nraise SystemExit(0)\n')
    probe.chmod(0o700)
    if subprocess.run([str(probe)], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                      stderr=subprocess.DEVNULL, timeout=2).returncode:
        raise ValueError('scratch execution probe failed')
    probe.unlink()
    return proof


def parent_dir(root, path, create=False):
    parts = relative_path(path).split('/')
    fd = os.dup(root)
    try:
        for part in parts[:-1]:
            if create:
                try:
                    os.mkdir(part, 0o700, dir_fd=fd)
                except FileExistsError:
                    pass
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        return fd, parts[-1]
    except BaseException:
        os.close(fd)
        raise


def load_tree(stream, cap):
    root = os.open('/scratch/work', os.O_RDONLY | os.O_DIRECTORY)
    seen = set()
    try:
        for metadata, data in records(stream, cap, 100000):
            if set(metadata) != {'path', 'mode', 'size'} or metadata['mode'] not in {'100644', '100755', '120000'}:
                raise ValueError('invalid tree inventory')
            path = relative_path(metadata['path'])
            if path in seen:
                raise ValueError('duplicate tree entry')
            seen.add(path)
            directory, leaf = parent_dir(root, path, create=True)
            try:
                if metadata['mode'] == '120000':
                    os.symlink(data.decode('utf8'), leaf, dir_fd=directory)
                else:
                    fd = os.open(leaf, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o700 if metadata['mode'] == '100755' else 0o600, dir_fd=directory)
                    with os.fdopen(fd, 'wb') as file:
                        file.write(data)
            finally:
                os.close(directory)
    finally:
        os.close(root)


def read_output(root, path, limit):
    directory, leaf = parent_dir(root, path)
    try:
        fd = os.open(leaf, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    finally:
        os.close(directory)
    with os.fdopen(fd, 'rb') as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > limit:
            raise ValueError('unsafe or oversized output file')
        data = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
        identity = lambda s: (s.st_dev, s.st_ino, s.st_mode, s.st_nlink, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        if identity(before) != identity(after) or len(data) != before.st_size:
            raise ValueError('output file changed during read')
        return data


def send(channel, value):
    channel.send(json.dumps(value).encode())


def receive(channel):
    data = channel.recv(65536)
    if not data:
        raise ValueError('host supervisor lost')
    return json.loads(data)


def run_phase(channel, name, command, timeout, head):
    if not command:
        return
    pid = os.fork()
    if pid == 0:
        try:
            # Kernel attaches host-namespace credentials to this datagram.
            send(channel, {'gate': name})
            if receive(channel) != {'release': name}:
                os._exit(125)
            gate = channel.fileno()
            os.close(gate)
            os.closerange(3, 1048576)
            null = os.open('/dev/null', os.O_RDONLY)
            os.dup2(null, 0)
            os.close(null)
            os.chdir('/scratch/work')
            os.execve('/bin/sh', ['/bin/sh', '-c', command], {
                'PATH': '/usr/bin:/bin', 'HOME': '/scratch/home', 'TMPDIR': '/scratch/tmp',
                'LC_ALL': 'C', 'AGENT_LOOP_REVIEW_IMAGES_HEAD_SHA': head,
                'AGENT_LOOP_REVIEW_IMAGES_MANIFEST': '/scratch/output/manifest.json'})
        except BaseException:
            os._exit(126)
    deadline = time.monotonic() + timeout
    code = None
    while time.monotonic() < deadline:
        got, status = os.waitpid(pid, os.WNOHANG)
        if got:
            code = os.waitstatus_to_exitcode(status)
            break
        time.sleep(.01)
    send(channel, {'phase_done': name, 'code': code})
    if receive(channel) != {'empty': name}:
        raise ValueError('payload emptiness not verified')
    if code is None:
        raise TimeoutError(name + ' timeout')
    if code:
        raise ValueError(name + ' nonzero exit: ' + str(code))


def main():
    # Prevent payload from reading supervisor memory/fds through proc or ptrace.
    if ctypes.CDLL(None).prctl(38, 1, 0, 0, 0) or ctypes.CDLL(None).prctl(4, 0, 0, 0, 0):
        raise ValueError('non-dumpable supervisor unavailable')
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    channel = socket.socket(fileno=int(sys.argv[1]))
    config = json.loads(Path('/trusted/config.json').read_bytes())
    if fcntl.fcntl(0, fcntl.F_GETFL) & os.O_ACCMODE != os.O_RDONLY:
        raise ValueError('input byte channel is not read-only')
    if config['mode'] == 'validate' and not stat.S_ISCHR(os.fstat(1).st_mode):
        raise ValueError('validator stdout is not a reviewed device')
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.unshare(0x10000000) == 0 or ctypes.get_errno() not in (errno.EPERM, errno.EACCES, errno.ENOSPC):
        raise ValueError('nested user namespaces are not disabled')
    proof = mount_probe(config['scratch_bytes'])
    proof['runtime_version'] = sys.version.split()[0]
    proof['input_read_only'] = True
    proof['nested_userns_disabled'] = True
    if config['mode'] == 'validate':
        proof['stdout_device'] = True
    proof['namespaces'] = {}
    for name, host in config['host_namespaces'].items():
        current = os.readlink('/proc/self/ns/' + name)
        if current == host:
            raise ValueError('unisolated namespace: ' + name)
        proof['namespaces'][name] = current
    send(channel, {'probe': proof})
    if receive(channel) != {'probe': 'accepted'}:
        raise ValueError('mount verification rejected')
    mode = config['mode']
    if mode == 'validate':
        resource.setrlimit(resource.RLIMIT_AS, (256 * 1024**2, 256 * 1024**2))
        resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
        limits = ImageLimits(**config['limits'])
        dimensions = {}
        for metadata, data in records(sys.stdin.buffer, limits.total_bytes, limits.entries * 2):
            if set(metadata) != {'path', 'size'} or metadata['path'] in dimensions or len(data) > limits.file_bytes:
                raise ValueError('invalid validator input')
            dimensions[relative_path(metadata['path'])] = validate_png(data, pixels=limits.pixels, dimension=limits.dimension)
        send(channel, {'validated': dimensions})
        return
    scratch_root = os.open('/scratch', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    root = os.open('output', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=scratch_root)
    output_identity = os.fstat(root)
    import hashlib
    class HashReader:
        digest = hashlib.sha256()
        def read(self, length):
            value = sys.stdin.buffer.read(length)
            self.digest.update(value)
            return value
    source = HashReader()
    load_tree(source, config['scratch_bytes'])
    if source.digest.hexdigest() != config['tree_digest']:
        raise ValueError('received tree inventory/content digest mismatch')
    send(channel, {'loaded': True})
    if receive(channel) != {'loaded': 'accepted'}:
        raise ValueError('input verification failed')
    # Payload diagnostics have no route onto trusted export stdout. Keep only
    # a bounded tail, in a draining thread, including during child waits.
    import threading
    read_fd, write_fd = os.pipe()
    tail = bytearray()
    def drain():
        while True:
            chunk = os.read(read_fd, 8192)
            if not chunk:
                return
            tail.extend(chunk)
            if len(tail) > 65536:
                del tail[:-65536]
    original_out, original_err = os.dup(1), os.dup(2)
    os.dup2(write_fd, 1)
    os.dup2(write_fd, 2)
    os.close(write_fd)
    thread = threading.Thread(target=drain, daemon=True)
    thread.start()
    try:
        run_phase(channel, 'setup', config['setup'], config['setup_timeout'], config['head'])
        run_phase(channel, 'capture', config['command'], config['capture_timeout'], config['head'])
    finally:
        os.dup2(original_out, 1)
        os.dup2(original_err, 2)
        os.close(original_out)
        os.close(original_err)
    diagnostic = bytes(tail).decode(errors='replace')
    for start in range(0, len(diagnostic), 8192):
        send(channel, {'diagnostic': diagnostic[start:start + 8192]})
    send(channel, {'export': 'ready'})
    if receive(channel) != {'export': 'accepted'}:
        raise ValueError('export not authorized')
    current_output = os.stat('output', dir_fd=scratch_root, follow_symlinks=False)
    if (current_output.st_dev, current_output.st_ino) != (output_identity.st_dev, output_identity.st_ino):
        raise ValueError('held output root was replaced')
    try:
        limits = ImageLimits(**config['limits'])
        raw = read_output(root, 'manifest.json', limits.manifest_bytes)
        _, paths = schema(raw, config['head'], limits)
        write_record(sys.stdout.buffer, {'path': 'manifest.json', 'size': len(raw)}, raw)
        total = 0
        for path in sorted(set(paths)):
            if path == 'manifest.json':
                raise ValueError('manifest cannot be an image')
            content = read_output(root, path, limits.file_bytes)
            total += len(content)
            if total > limits.total_bytes:
                raise ValueError('export byte limit exceeded')
            write_record(sys.stdout.buffer, {'path': path, 'size': len(content)}, content)
        write_record(sys.stdout.buffer, {'complete': True})
    finally:
        os.close(root)
        os.close(scratch_root)


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        # Only bounded trusted diagnostics; payload stderr never becomes framing.
        print(type(exc).__name__ + ': ' + str(exc)[:2048], file=sys.stderr)
        sys.exit(1)
