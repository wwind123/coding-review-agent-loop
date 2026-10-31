"""Pure validation of the complete sandbox mount inventory."""
import math

DEVICES = {'/dev/null', '/dev/zero', '/dev/full', '/dev/random', '/dev/urandom'}


def verify_mounts(mountinfo, capacity, size, page):
    mounts = {}
    for line in mountinfo.splitlines():
        before, after = line.split(' - ', 1)
        fields = before.split()
        target, options = fields[4], set(fields[5].split(','))
        fs, _, superoptions = after.split()[:3]
        if target in mounts:
            raise ValueError('overlapping sandbox mount')
        mounts[target] = (fs, options, superoptions)
        if 'rw' in options and target != '/scratch' and target not in DEVICES:
            raise ValueError('unexpected writable storage mount: ' + target)
    for path in ('/', '/proc', '/dev'):
        if path not in mounts or 'ro' not in mounts[path][1]:
            raise ValueError('required read-only mount: ' + path)
    if mounts['/dev'][0] != 'tmpfs' or mounts['/proc'][0] != 'proc':
        raise ValueError('unexpected device/proc mount type')
    fs, options, _ = mounts['/scratch']
    expected = ((size + page - 1) // page) * page
    if (fs != 'tmpfs' or not {'nodev', 'nosuid', 'rw'} <= options or 'noexec' in options or capacity != expected):
        raise ValueError('unverified scratch type/options/size')
    return {'mounts': {k: [v[0], sorted(v[1]), v[2]] for k, v in mounts.items()}, 'scratch_bytes': expected}
