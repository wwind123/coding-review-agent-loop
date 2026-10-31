"""Bounded binary records for trusted staging and output channels."""
import json
import os
import struct


def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def loads(data):
    return json.loads(data, object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def exact(stream, length):
    parts = []
    while length:
        part = stream.read(min(length, 65536))
        if not part:
            raise ValueError('truncated transfer')
        parts.append(part)
        length -= len(part)
    return b''.join(parts)


def write_record(stream, metadata, content=b''):
    header = json.dumps(metadata, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()
    if len(header) > 8192:
        raise ValueError('oversized transfer header')
    stream.write(struct.pack('>I', len(header)))
    stream.write(header)
    stream.write(content)
    stream.flush()


def records(stream, max_bytes, max_entries):
    total = count = 0
    while True:
        length = struct.unpack('>I', exact(stream, 4))[0]
        if not 0 < length <= 8192:
            raise ValueError('invalid transfer header')
        metadata = loads(exact(stream, length))
        if metadata == {'complete': True}:
            if stream.read(1):
                raise ValueError('trailing transfer bytes')
            return
        if not isinstance(metadata, dict):
            raise ValueError('invalid transfer record')
        size = metadata.get('size')
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError('invalid transfer size')
        total += size
        count += 1
        if total > max_bytes or count > max_entries:
            raise ValueError('transfer limit exceeded')
        yield metadata, exact(stream, size)
