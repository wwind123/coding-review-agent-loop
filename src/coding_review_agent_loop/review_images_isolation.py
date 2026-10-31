"""Fail-closed Bubblewrap layouts; only root-owned offline runtime grants."""
from __future__ import annotations
import fcntl
import os
from pathlib import Path
import re
import subprocess
import sys

from .secure_git import _trusted_path


def sealed(data: bytes, name='capture-control'):
    fd = os.memfd_create(name, os.MFD_ALLOW_SEALING | os.MFD_CLOEXEC)
    try:
        with os.fdopen(os.dup(fd), 'wb') as stream:
            stream.write(data)
        fcntl.fcntl(fd, fcntl.F_ADD_SEALS, fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL)
        os.lseek(fd, 0, os.SEEK_SET)
        return fd
    except BaseException:
        os.close(fd)
        raise


def preflight(grants):
    if sys.platform != 'linux':
        raise ValueError('capture isolation requires Linux')
    bwrap = _trusted_path('/usr/bin/bwrap', executable=True)
    env = {'PATH': '/usr/bin:/bin', 'LC_ALL': 'C'}
    version = subprocess.run([bwrap, '--version'], env=env, stdin=subprocess.DEVNULL, capture_output=True, timeout=5)
    match = re.search(rb'(\d+)\.(\d+)\.(\d+)', version.stdout)
    if version.returncode or not match or tuple(map(int, match.groups())) < (0, 9, 0):
        raise ValueError('capture requires Bubblewrap >= 0.9.0')
    help_result = subprocess.run([bwrap, '--help'], env=env, stdin=subprocess.DEVNULL, capture_output=True, timeout=5)
    if help_result.returncode or b'--size' not in help_result.stdout:
        raise ValueError('Bubblewrap lacks --size')
    if not grants:
        raise ValueError('explicit offline runtime grants are required')
    normalized = []
    for path in grants:
        # Permit specific system runtimes/assets, never whole installation
        # prefixes (which can also contain operator configuration/secrets).
        canonical = os.path.realpath(path)
        runtime_roots = ('/usr/bin', '/usr/sbin', '/usr/lib', '/usr/lib64', '/usr/libexec',
                         '/usr/local/bin', '/usr/local/lib', '/usr/local/lib64')
        specific_asset = canonical.startswith(('/usr/share/', '/usr/local/share/'))
        specific_product = canonical.startswith('/opt/') and len(Path(canonical).parts) >= 4
        if not (specific_asset or specific_product or any(canonical == prefix or canonical.startswith(prefix + '/') for prefix in runtime_roots)):
            raise ValueError('unsafe runtime grant')
        if '..' in Path(path).parts or str(Path(path)) != path:
            raise ValueError('noncanonical runtime grant')
        resolved = _trusted_path(path, executable=False) if not Path(path).is_dir() else trusted_directory(path)
        if any(resolved == old or resolved.startswith(old + '/') or old.startswith(resolved + '/') for old in normalized):
            raise ValueError('overlapping runtime grants')
        normalized.append(resolved)
    return bwrap


def trusted_directory(path):
    resolved = os.path.realpath(path)
    for candidate in (Path(path), Path(resolved)):
        for component in (candidate, *candidate.parents):
            info = component.lstat()
            if info.st_uid or (not component.is_symlink() and info.st_mode & 0o022):
                raise ValueError('runtime grant must be root-owned and nonwritable')
    return resolved


def argv(bwrap, grants, scratch_bytes, files, *, gate_fd):
    result = [bwrap, '--unshare-all', '--unshare-user', '--disable-userns', '--assert-userns-disabled', '--unshare-pid', '--unshare-ipc', '--unshare-uts', '--unshare-net', '--die-with-parent', '--new-session', '--as-pid-1',
              '--cap-drop', 'ALL', '--clearenv', '--setenv', 'PATH', '/usr/bin:/bin',
              '--setenv', 'HOME', '/scratch/home', '--setenv', 'TMPDIR', '/scratch/tmp',
              '--setenv', 'LC_ALL', 'C', '--proc', '/proc',
              '--size', '65536', '--tmpfs', '/dev',
              '--dir', '/trusted', '--size', str(scratch_bytes), '--tmpfs', '/scratch',
              '--symlink', '/scratch/tmp', '/tmp', '--symlink', '/scratch/tmp', '/var/tmp',
              '--symlink', '/scratch/shm', '/dev/shm']
    for device in ('null', 'zero', 'full', 'random', 'urandom'):
        result += ['--dev-bind', '/dev/' + device, '/dev/' + device]
    result += ['--symlink', '/proc/self/fd', '/dev/fd']
    result += ['--dir', '/usr/bin', '--dir', '/usr/lib', '--dir', '/usr/lib64']
    # Construct usr-merge aliases before creating destinations below them.
    for path in ('/bin', '/lib', '/lib64'):
        if path not in grants and os.path.islink(path):
            result += ['--symlink', os.readlink(path), path]
    for path in grants:
        result += ['--ro-bind', os.path.realpath(path), path]
    for name, fd in files.items():
        result += ['--ro-bind-data', str(fd), '/trusted/' + name]
    result += ['--remount-ro', '/proc', '--remount-ro', '/dev', '--remount-ro', '/',
               '--', '/usr/bin/python3', '-I', '/trusted/review_images_worker.py', str(gate_fd)]
    return result


def validator_grants(operator_grants):
    """Narrow the operator allowlist to trusted Python/stdlib and their ELF libs.

    Inspect only the root-owned system interpreter and its standard-library
    extensions. Never run ldd against repository input or accept repo paths.
    """
    import json
    python = _trusted_path('/usr/bin/python3', executable=True)
    probe = subprocess.run([python, '-I', '-S', '-c',
                            'import sysconfig,json; print(json.dumps(sysconfig.get_path("stdlib")))'],
                           env={'PATH': '/usr/bin:/bin', 'LC_ALL': 'C'}, stdin=subprocess.DEVNULL,
                           capture_output=True, timeout=5, check=True)
    stdlib = trusted_directory(json.loads(probe.stdout))
    extensions = sorted(Path(stdlib, 'lib-dynload').glob('*.so'))
    if len(extensions) > 256:
        raise ValueError('unexpected Python runtime layout')
    targets = [python, *[_trusted_path(str(path), executable=False) for path in extensions]]
    ldd = _trusted_path('/usr/bin/ldd', executable=True)
    dependencies = subprocess.run([ldd, *targets], env={'PATH': '/usr/bin:/bin', 'LC_ALL': 'C'},
                                  stdin=subprocess.DEVNULL, capture_output=True, timeout=10, check=True)
    if len(dependencies.stdout) > 256*1024:
        raise ValueError('oversized runtime dependency inventory')
    candidates = ['/usr/bin/python3', stdlib]
    for line in dependencies.stdout.decode().splitlines():
        if 'not found' in line:
            raise ValueError('Python native runtime is unavailable')
        match = re.search(r'(?:=>\s*)?(/[^\s]+)\s+\(', line)
        if match:
            candidates.append(match.group(1))
    allowed = [os.path.realpath(path) for path in operator_grants]
    selected = {}
    # Preserve the loader's expected alias; /lib and /lib64 links are recreated
    # inside the sandbox but no broad native-library directory is exposed.
    for path in candidates:
        real = _trusted_path(path, executable=False) if not Path(path).is_dir() else trusted_directory(path)
        if not any(real == grant or real.startswith(grant + '/') for grant in allowed):
            raise ValueError('operator grants do not cover isolated Python runtime')
        if real not in selected or path.startswith('/lib64/'):
            selected[real] = path
    result = tuple(selected.values())
    preflight(result)
    return result
