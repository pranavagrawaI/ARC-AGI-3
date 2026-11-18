"""Read ZIP directory ranges only; never download exported file contents."""
import json
from pathlib import Path
import struct
import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiDownloadKernelOutputRequest

BASE = Path(__file__).resolve().parents[1] / 'actor-critic-construction-kaggle-20260914T195518Z'

def main():
    api = KaggleApi()
    api.authenticate()
    request = ApiDownloadKernelOutputRequest()
    request.owner_slug = 'pranavagrawai'
    request.kernel_slug = 'actor-critic-construction-kaggle-r4'
    request.version_number = 1
    with api.build_kaggle_client() as client:
        response = client.kernels.kernels_api_client.download_kernel_output(request)
        url = response.url
        response.close()
    ranges = []
    def fetch(spec, cap):
        with requests.get(url, headers={'Range': spec}, stream=True, timeout=60) as r:
            if r.status_code != 206:
                raise RuntimeError(f'range unavailable: HTTP {r.status_code}')
            data = bytearray()
            for chunk in r.iter_content(65536):
                data.extend(chunk)
                if len(data) > cap:
                    raise RuntimeError('range exceeds bound')
            ranges.append({'range': spec, 'content_range': r.headers.get('Content-Range'), 'bytes': len(data)})
            return bytes(data)
    end = fetch('bytes=-65557', 65557)
    pos = end.rfind(b'PK\x05\x06')
    if pos < 0:
        raise RuntimeError('ZIP end record missing')
    eocd = struct.unpack_from('<4s4H2IH', end, pos)
    count, size, offset = eocd[4:7]
    if count == 65535 or size == 4294967295 or offset == 4294967295:
        locator = struct.unpack_from('<4sIQI', end, pos - 20)
        if locator[0] != b'PK\x06\x07':
            raise RuntimeError('ZIP64 locator missing')
        z = fetch(f'bytes={locator[2]}-{locator[2]+55}', 56)
        record = struct.unpack('<4sQHHIIQQQQ', z)
        if record[0] != b'PK\x06\x06':
            raise RuntimeError('ZIP64 end record missing')
        count, size, offset = record[7:10]
    if size > 64 * 1024 * 1024:
        raise RuntimeError('directory exceeds 64MiB bound')
    directory = fetch(f'bytes={offset}-{offset+size-1}', size)
    names = []
    cursor = 0
    while cursor < len(directory):
        header = struct.unpack_from('<4s6H3I5H2I', directory, cursor)
        if header[0] != b'PK\x01\x02':
            raise RuntimeError('bad directory record')
        length, extra, comment = header[10:13]
        name = directory[cursor+46:cursor+46+length].decode('utf-8')
        names.append(name)
        cursor += 46 + length + extra + comment
    if len(names) != count or cursor != size:
        raise RuntimeError('directory count mismatch')
    selected = [n for n in names if n.startswith('model-runtime/solver-workspaces/raw/workspaces/') and not any(p.startswith('.') or 'credential' in p.lower() for p in Path(n).parts)]
    result = {'requested_version': 1, 'total_entries_verified': count, 'ranges': ranges, 'workspace_names': selected, 'file_contents_downloaded': False}
    with (BASE / 'ZIP-DIRECTORY-READBACK.json').open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps({'total_entries_verified': count, 'workspace_names': selected}, indent=2))

if __name__ == '__main__':
    main()
