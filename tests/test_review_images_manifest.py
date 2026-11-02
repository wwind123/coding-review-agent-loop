import dataclasses
import io
import json
import os
import sys
import struct
import zlib

import pytest

if sys.platform != 'linux':
    if os.environ.get('AGENT_LOOP_CAPTURE_SECURITY_REQUIRED') == '1':
        pytest.fail('required capture security evidence needs Linux', pytrace=False)
    pytest.skip('capture filesystem/security cases require Linux', allow_module_level=True)
from coding_review_agent_loop.review_images_channel import records, write_record
from coding_review_agent_loop.review_images_config import ImageLimits
from coding_review_agent_loop.review_images_manifest import schema, publish
from coding_review_agent_loop.review_images_png import validate_png
from review_images_helpers import chunk, png, manifest


@pytest.mark.parametrize('color,depth', [(c,d) for c,ds in {0:(1,2,4,8,16),2:(8,16),3:(1,2,4,8),4:(8,16),6:(8,16)}.items() for d in ds])
def test_legal_png_formats(color, depth):
    assert validate_png(png(width=3, color=color, depth=depth)) == (3, 2)


def malformed():
    valid = png()
    crc = bytearray(valid)
    crc[29] ^= 1
    return [b'not png', bytes(crc), valid[:-2], png(width=16001), png(raw=b'\0' * 100000), valid+b'trailing',
            valid[:-12], png(compressed=zlib.compress(bytes(18))+b'trail'), png(compressed=zlib.compress(bytes(18))[:-2]),
            png(raw=bytes(17)), png(raw=bytes(19)), png(raw=b'\5'+bytes(17)), png(interlace=1),
            png(depth=3), png(color=1), png(before=chunk(b'acTL')), png(after=chunk(b'fcTL')),
            png(after=chunk(b'fdAT')), png(after=chunk(b'tEXt')+chunk(b'IDAT', b'x')),
            png(before=chunk(b'PLTE', b'x')), png(before=chunk(b'IHDR', bytes(13))),
            png(before=chunk(b'ABCD')), png(compressed=zlib.compress(bytes(18))+zlib.compress(bytes(18)))]


@pytest.mark.parametrize('data', malformed(), ids=[f'malformed-{n}' for n in range(len(malformed()))])
def test_reject_malformed_png(data):
    with pytest.raises((ValueError, zlib.error)):
        validate_png(data)


def test_crc_checked_for_every_chunk_class():
    for kind, payload in [(b'tEXt', b'hello'), (b'IDAT', zlib.compress(bytes(18))), (b'IEND', b''), (b'PLTE', bytes(3))]:
        broken = bytearray(chunk(kind, payload))
        broken[-1] ^= 1
        with pytest.raises(ValueError, match='CRC'):
            validate_png(png(before=bytes(broken)))


def test_consecutive_idat_exact_scanline_bound():
    data = png()
    compressed = zlib.compress(bytes(18))
    split = data[:33]+chunk(b'IDAT', compressed[:3])+chunk(b'IDAT', compressed[3:])+chunk(b'IEND')
    assert validate_png(split, pixels=4, dimension=2) == (2, 2)
    with pytest.raises(ValueError):
        validate_png(split, pixels=3)


def test_manifest_and_immutable_snapshot_lifetime():
    raw = json.dumps(manifest()).encode()
    data = png()
    snapshot = publish(raw, 'a'*40, ImageLimits(), {'one.png':data,'base.png':data},
                       {'one.png':(2,2),'base.png':(2,2)}, 'config')
    with snapshot:
        assert snapshot.files[0].read() == data
        with pytest.raises(OSError):
            os.write(snapshot.files[0].fd, b'x')
        changed = publish(raw, 'a'*40, ImageLimits(), {'one.png':data,'base.png':data},
                          {'one.png':(2,2),'base.png':(2,2)}, 'other')
        assert snapshot.digest != changed.digest
        changed.close()
    with pytest.raises(OSError):
        snapshot.files[0].read()


@pytest.mark.parametrize('mutation', ['head', 'empty', 'duplicate', 'viewport', 'unsafe', 'baseline', 'score', 'control'])
def test_manifest_schema_rejects_bad_metadata(mutation):
    value = manifest()
    image = value['images'][0]
    if mutation == 'head': value['head_sha'] = 'b'*40
    if mutation == 'empty': value['images'] = []
    if mutation == 'duplicate': value['images'].append(dict(image))
    if mutation == 'viewport': image['viewport']['width'] = True
    if mutation == 'unsafe': image['path'] = '../secret'
    if mutation == 'baseline': image['baseline_path'] = '/etc/passwd'
    if mutation == 'score': image['diff_score'] = float('nan')
    if mutation == 'control': image['state'] = 'bad\nstate'
    with pytest.raises(ValueError):
        schema(json.dumps(value).encode(), 'a'*40, ImageLimits())


def test_duplicate_json_keys_and_framing():
    with pytest.raises(ValueError):
        schema(b'{"head_sha":"a","head_sha":"b","images":[]}', 'a', ImageLimits())
    stream = io.BytesIO()
    write_record(stream, {'path':'image','size':3}, b'png')
    write_record(stream, {'complete':True})
    assert list(records(io.BytesIO(stream.getvalue()),3,1))[0][1] == b'png'
    for data in (stream.getvalue()[:-1], stream.getvalue()+b'x', struct.pack('>I',9000)):
        with pytest.raises(ValueError):
            list(records(io.BytesIO(data),3,1))


@pytest.mark.parametrize('variant',['compression','filtering','missing-idat','ihdr-not-first','iend-payload','indexed-no-palette','indexed-excess-palette','gray-palette'])
def test_png_remaining_structure_rules(variant):
    data=png()
    if variant in {'compression','filtering'}:
        header=bytearray(data[16:29])
        header[10 if variant=='compression' else 11]=1
        data=data[:8]+chunk(b'IHDR',bytes(header))+data[33:]
    if variant=='missing-idat': data=data[:33]+chunk(b'IEND')
    if variant=='ihdr-not-first': data=data[:8]+chunk(b'tEXt')+data[8:]
    if variant=='iend-payload': data=data[:-12]+chunk(b'IEND',b'x')
    if variant=='indexed-no-palette': data=png(color=3,before=chunk(b'tEXt'))
    if variant=='indexed-excess-palette': data=png(color=3,depth=1,before=chunk(b'PLTE',bytes(9)))
    if variant=='gray-palette': data=png(color=0,before=chunk(b'PLTE',bytes(3)))
    with pytest.raises(ValueError): validate_png(data)


@pytest.mark.parametrize('limit',['file_bytes','total_bytes','pixels','total_pixels','dimension'])
def test_snapshot_limits_count_baseline_references(limit):
    data=png()
    raw=json.dumps(manifest()).encode()
    limits=dataclasses.replace(ImageLimits(),**{limit:1})
    with pytest.raises(ValueError):
        publish(raw,'a'*40,limits,{'one.png':data,'base.png':data},{'one.png':(2,2),'base.png':(2,2)},'config')


def test_close_is_idempotent_when_descriptor_number_is_reused():
    data=png()
    snapshot=publish(json.dumps(manifest()).encode(),'a'*40,ImageLimits(),{'one.png':data,'base.png':data},
                     {'one.png':(2,2),'base.png':(2,2)},'config')
    snapshot.close()
    fd=os.open('/dev/null',os.O_RDONLY)
    try:
        snapshot.close()
        os.fstat(fd)
        with pytest.raises(OSError): snapshot.files[0].read()
    finally: os.close(fd)


def test_manifest_rejects_unicode_controls_and_overflowing_score():
    for field,value in [('locale','en\x85'),('state','ready\u202e'),('diff_score',10**1000)]:
        document=manifest()
        document['images'][0][field]=value
        with pytest.raises(ValueError): schema(json.dumps(document).encode(),'a'*40,ImageLimits())


def test_malformed_validator_reply_cannot_publish():
    with pytest.raises(ValueError,match='reply'):
        publish(json.dumps(manifest()).encode(),'a'*40,ImageLimits(),{},None,'config')


@pytest.mark.parametrize('viewport', [(390, 844), (1280, 720), (16000, 16000)])
def test_viewport_metadata_is_independent_of_downscaled_png_bounds(viewport):
    from coding_review_agent_loop.review_images_config import delivery_limits
    value = manifest()
    value['images'][0]['viewport'] = dict(zip(('width', 'height'), viewport))
    data = png()
    with publish(json.dumps(value).encode(), 'a'*40, delivery_limits(),
                 {'one.png': data, 'base.png': data},
                 {'one.png': (2, 2), 'base.png': (2, 2)}, 'config') as snapshot:
        assert json.loads(snapshot.manifest_json)['images'][0]['viewport']['height'] == viewport[1]


@pytest.mark.parametrize('dimension', [16001, 0, -1, True, 1.5])
def test_viewport_metadata_has_its_own_strict_bound(dimension):
    from coding_review_agent_loop.review_images_config import delivery_limits
    value = manifest()
    value['images'][0]['viewport']['height'] = dimension
    with pytest.raises(ValueError, match='viewport'):
        schema(json.dumps(value).encode(), 'a'*40, delivery_limits())
