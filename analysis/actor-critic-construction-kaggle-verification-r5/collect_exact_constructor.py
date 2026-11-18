"""Bounded exact-version, known-file retrieval without cache enumeration."""
import hashlib
import json
from pathlib import Path
import time
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiDownloadKernelOutputRequest

BASE = Path(__file__).resolve().parents[1] / 'actor-critic-construction-kaggle-20260914T235215Z'
FILES = ('model-runtime/solver-workspaces/raw/logical-requests.jsonl', 'model-runtime/solver-workspaces/raw/requests.jsonl', 'model-runtime/solver-workspaces/raw/responses.jsonl', 'model-runtime/solver-workspaces/raw/latency.jsonl', 'model-runtime/solver-workspaces/raw/raw-http/18.body', 'model-runtime/solver-workspaces/raw/raw-http/19.body')

def main():
    api = KaggleApi()
    api.authenticate()
    rows = []
    with api.build_kaggle_client() as client:
        for name in FILES:
            request = ApiDownloadKernelOutputRequest()
            request.owner_slug = 'pranavagrawai'
            request.kernel_slug = 'actor-critic-construction-kaggle-r5'
            request.file_path = name
            request.version_number = 1
            row = {'file': name, 'version_number_requested': 1, 'started_unix': time.time()}
            try:
                response = client.kernels.kernels_api_client.download_kernel_output(request)
                row['http_status'] = response.status_code
                raw = bytearray()
                for chunk in response.iter_content(65536):
                    raw.extend(chunk)
                    if len(raw) > 8 * 1024 * 1024:
                        raise RuntimeError('file exceeds evidence bound')
                response.close()
                if name.endswith(".jsonl"):
                    for line in raw.splitlines():
                        if line.strip(): json.loads(line)
                else:
                    json.loads(raw)
                dest = BASE / 'runtime-evidence-exact-request' / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                with dest.open('xb') as handle:
                    handle.write(raw)
                row.update(status='RETRIEVED', sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
            except Exception as exc:
                row.update(status='FAILED', error_type=type(exc).__name__)
                resp = getattr(exc, 'response', None)
                if resp is not None:
                    row['http_status'] = resp.status_code
            rows.append(row)
            (BASE / 'direct-exact-constructor-receipts.json').write_text(json.dumps(rows, indent=2) + '\n')
            if row.get('http_status') == 429:
                break
    print(json.dumps(rows, indent=2))

if __name__ == '__main__':
    main()
