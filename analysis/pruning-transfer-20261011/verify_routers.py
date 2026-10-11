"""Verify retained expert indices against pinned Intel and RadixArk router bytes.

Range reads only; never downloads a whole checkpoint shard.
"""
import concurrent.futures
import hashlib
import json
import struct
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE / 'checkpoints'


def ranged(url, start, end):
    request = urllib.request.Request(url + f'?range_start={start}&range_end={end}',
                                     headers={'Range': f'bytes={start}-{end}'})
    with urllib.request.urlopen(request, timeout=45) as response:
        if response.status != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {start}-{end}/'):
            raise ValueError('Range request not honored')
        data = response.read(end - start + 2)
    if len(data) != end - start + 1:
        raise ValueError('Truncated or oversized range')
    return data


def tensor_header(url):
    size = struct.unpack('<Q', ranged(url, 0, 7))[0]
    if not 0 < size < 64 * 1024 * 1024:
        raise ValueError('Unexpected safetensors header size')
    return 8 + size, json.loads(ranged(url, 8, 7 + size))


def main():
    identities = {key: json.loads((CACHE / f'{key}-identity.json').read_text()) for key in ['intel', 'radix']}
    indices = {key: json.loads((CACHE / f'{key}-model.safetensors.index.json').read_text())['weight_map'] for key in identities}
    names = [f'model.language_model.layers.{layer}.mlp.gate.weight' for layer in range(48)]
    urls = {(key, indices[key][name]): f"https://huggingface.co/{identity['repo']}/resolve/{identity['revision']}/{indices[key][name]}"
            for key, identity in identities.items() for name in names}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        headers = dict(zip(urls, pool.map(tensor_header, urls.values())))

        def check(name):
            arrays = {}
            for key in identities:
                shard = indices[key][name]
                base, header = headers[key, shard]
                meta = header[name]
                if meta['dtype'] != 'BF16' or meta['shape'] != [512, 2560]:
                    raise ValueError((name, key, meta))
                start, end = meta['data_offsets']
                arrays[key] = ranged(urls[key, shard], base + start, base + end - 1)
            return dict(name=name, exact_equal=arrays['intel'] == arrays['radix'],
                        sha256={k: hashlib.sha256(v).hexdigest() for k, v in arrays.items()})

        results = list(pool.map(check, names))
    report = dict(identities=identities, routers=results, all_equal=all(r['exact_equal'] for r in results))
    (HERE / 'router-identity.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'all_equal': report['all_equal'], 'routers': len(results)}))
    if not report['all_equal']:
        raise SystemExit('Router correspondence differs; do not transplant mask')


if __name__ == '__main__':
    main()
