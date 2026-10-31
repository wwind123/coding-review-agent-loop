"""Bounded standard-library PNG structural validator, also loaded in isolation.

This does not reconstruct pixels or compare baselines. No filesystem access.
"""
import struct
import zlib

_SIGNATURE = b'\x89PNG\r\n\x1a\n'
_DEPTHS = {0: (1, 2, 4, 8, 16), 2: (8, 16), 3: (1, 2, 4, 8), 4: (8, 16), 6: (8, 16)}
_SAMPLES = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}


def validate_png(data: bytes, *, pixels=16_000_000, dimension=16_000) -> tuple[int, int]:
    if data[:8] != _SIGNATURE:
        raise ValueError('invalid PNG signature')
    offset = 8
    ihdr = palette = ended = False
    idat = 0
    idat_closed = False
    total = 0
    decoder = None
    width = height = row_size = expected = depth = color = 0
    while offset < len(data):
        if len(data) - offset < 12:
            raise ValueError('truncated PNG chunk')
        length, kind = struct.unpack_from('>I4s', data, offset)
        if length > 0x7fffffff or length > len(data) - offset - 12:
            raise ValueError('invalid PNG chunk length')
        if any(not (65 <= c <= 90 or 97 <= c <= 122) for c in kind) or kind[2] & 32:
            raise ValueError('invalid PNG chunk type')
        payload = memoryview(data)[offset + 8:offset + 8 + length]
        crc = struct.unpack_from('>I', data, offset + 8 + length)[0]
        if zlib.crc32(payload, zlib.crc32(kind)) != crc:
            raise ValueError('PNG CRC mismatch')
        offset += length + 12
        if not ihdr and kind != b'IHDR':
            raise ValueError('IHDR must be first')
        if kind in (b'acTL', b'fcTL', b'fdAT'):
            raise ValueError('APNG is unsupported')
        if kind == b'IHDR':
            if ihdr or length != 13:
                raise ValueError('invalid IHDR')
            width, height, depth, color, compression, filtering, interlace = struct.unpack('>IIBBBBB', payload)
            if (not 0 < width <= dimension or not 0 < height <= dimension or width * height > pixels
                    or color not in _DEPTHS or depth not in _DEPTHS[color]
                    or compression or filtering or interlace):
                raise ValueError('unsupported or oversized IHDR')
            row_size = 1 + (width * _SAMPLES[color] * depth + 7) // 8
            expected = height * row_size
            ihdr = True
            decoder = zlib.decompressobj()
        elif kind == b'PLTE':
            if palette or idat or color in (0, 4) or not length or length % 3 or length > 768:
                raise ValueError('invalid PLTE')
            if color == 3 and length // 3 > 2**depth:
                raise ValueError('palette exceeds indexed depth')
            palette = True
        elif kind == b'IDAT':
            if idat_closed or (color == 3 and not palette):
                raise ValueError('invalid IDAT order')
            idat += 1
            # Limit input slices and output allocations independently. Never flush
            # without a bound, including when exactly expected bytes were seen.
            for start in range(0, length, 32768):
                pending = bytes(payload[start:start + 32768])
                while pending:
                    if decoder.eof:
                        raise ValueError('trailing compressed input')
                    output = decoder.decompress(pending, min(32768, expected - total + 1))
                    if total + len(output) > expected:
                        raise ValueError('PNG decompression overflow')
                    first_filter = (-total) % row_size
                    if any(output[i] > 4 for i in range(first_filter, len(output), row_size)):
                        raise ValueError('invalid scanline filter')
                    total += len(output)
                    pending = decoder.unconsumed_tail
                    if decoder.unused_data:
                        raise ValueError('trailing compressed data')
        elif kind == b'IEND':
            if length or not idat or offset != len(data):
                raise ValueError('invalid IEND or trailing file data')
            ended = True
        elif not kind[0] & 32:
            raise ValueError('unknown critical PNG chunk')
        if idat and kind != b'IDAT':
            idat_closed = True
        if ended:
            break
    if not ended or not idat or total != expected or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise ValueError('incomplete PNG stream')
    return width, height
