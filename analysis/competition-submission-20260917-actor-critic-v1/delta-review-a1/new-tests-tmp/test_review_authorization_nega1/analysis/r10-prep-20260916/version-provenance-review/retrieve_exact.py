"""Read-only, bounded exact numeric-version retrieval; no listing or fallback."""
import hashlib
import json
import time
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiDownloadKernelOutputRequest

BASE = Path(__file__).resolve().parent
FILES = ('model-runtime/official-launcher-result.json', 'actor-critic-construction/SEAL.json')
LIMIT = 2 * 1024 * 1024

def main():
    # Refuse to overwrite a previous observation.
    out = BASE / ('exact-observation-' + str(time.time_ns()))
    out.mkdir()
    api = KaggleApi()
    api.authenticate()
    rows = []
    with api.build_kaggle_client() as client:
        for name in FILES:
            fields = dict(owner_slug='pranavagrawai', kernel_slug='r10a-cd82-constructor-repair', file_path=name, version_number=1)
            request = ApiDownloadKernelOutputRequest()
            for key, value in fields.items():
                setattr(request, key, value)
            row = dict(request_type='ApiDownloadKernelOutputRequest', service='kernels.KernelsApiService', method='DownloadKernelOutput', request=fields, started_unix=time.time())
            rows.append(row)
            (out / 'requests-responses.json').write_text(json.dumps(rows, indent=2) + '\n')
            response = None
            try:
                response = client.kernels.kernels_api_client.download_kernel_output(request)
                row['http_status'] = response.status_code
                response.raise_for_status()
                raw = bytearray()
                for chunk in response.iter_content(65536):
                    raw.extend(chunk)
                    if len(raw) > LIMIT:
                        raise RuntimeError('receipt exceeds 2 MiB bound')
                json.loads(raw)
                dest = out / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                with dest.open('xb') as handle:
                    handle.write(raw)
                row.update(status='RETRIEVED', response_file=str(dest.relative_to(BASE)), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
            except Exception as exc:
                row.update(status='FAILED', error_type=type(exc).__name__)
                resp = getattr(exc, 'response', None)
                if resp is not None:
                    row['http_status'] = resp.status_code
            finally:
                if response is not None:
                    response.close()
                row['finished_unix'] = time.time()
                (out / 'requests-responses.json').write_text(json.dumps(rows, indent=2) + '\n')
            if row.get('http_status') == 429:
                break
    print(json.dumps({'evidence_directory': str(out), 'requests': rows}, indent=2))

if __name__ == '__main__':
    main()
