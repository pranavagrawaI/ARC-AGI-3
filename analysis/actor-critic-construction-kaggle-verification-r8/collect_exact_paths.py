"""Bounded exact-version retrieval of named files (argv), never a tree listing. Read-only."""
import hashlib
import json
import sys
import time
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiDownloadKernelOutputRequest

BASE = Path(__file__).resolve().parents[1] / 'actor-critic-construction-kaggle-r8-gameplay-20260915T124458Z'
LIMIT = 16 * 1024 * 1024


def main(files, label):
    api = KaggleApi()
    api.authenticate()
    rows = []
    with api.build_kaggle_client() as client:
        for name in files:
            request = ApiDownloadKernelOutputRequest()
            request.owner_slug = 'pranavagrawai'
            request.kernel_slug = 'actor-critic-construction-kaggle-r8-gameplay'
            request.file_path = name
            request.version_number = 1
            row = {'file': name, 'version_number_requested': 1, 'started_unix': time.time()}
            try:
                response = client.kernels.kernels_api_client.download_kernel_output(request)
                row['http_status'] = response.status_code
                raw = bytearray()
                for chunk in response.iter_content(65536):
                    raw.extend(chunk)
                    if len(raw) > LIMIT:
                        raise RuntimeError('file exceeds evidence bound')
                response.close()
                dest = BASE / 'runtime-evidence-exact-request' / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists():
                    row.update(status='ALREADY_PRESENT', sha256=hashlib.sha256(dest.read_bytes()).hexdigest(), bytes=dest.stat().st_size)
                else:
                    with dest.open('xb') as handle:
                        handle.write(raw)
                    row.update(status='RETRIEVED', sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
            except Exception as exc:
                row.update(status='FAILED', error_type=type(exc).__name__)
                resp = getattr(exc, 'response', None)
                if resp is not None:
                    row['http_status'] = resp.status_code
            rows.append(row)
            (BASE / f'direct-exact-{label}-receipts.json').write_text(json.dumps(rows, indent=2) + '\n')
            if row.get('http_status') == 429:
                break
            time.sleep(0.5)
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main(sys.argv[2:], sys.argv[1])
