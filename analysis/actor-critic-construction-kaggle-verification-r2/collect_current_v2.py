"""Read-only current-identity evidence collector; never downloads unknown assets."""
import hashlib
import json
from pathlib import Path
import urllib.request
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

BASE = Path(__file__).resolve().parents[1] / 'actor-critic-construction-kaggle-20260914T183029Z'
TARGET = 'pranavagrawai/actor-critic-construction-kaggle-r2'
NAMES = {'GPU.json', 'SEAL.json', 'PROCESS-GUARD.json', 'BOOTSTRAP-FAILURE.json', 'result.json', 'probe-config.json', 'system-before.json', 'system-after.json', 'shutdown.json', 'lifecycle.json', 'event-log.jsonl', 'server-environment.json', 'server-command.json', 'runtime-verification.json', 'model-verification.json', 'RESULTS.json'}

def main():
    api = KaggleApi()
    api.authenticate()
    request = ApiListKernelSessionOutputRequest()
    request.user_name, request.kernel_slug = TARGET.split('/')
    request.page_size = 200
    rows, retained = [], []
    progress = BASE / 'runtime-evidence-retrieval-v2-progress.json'
    if progress.exists():
        saved = json.loads(progress.read_text())
        rows, retained = saved['files'], saved['retained']
        request.page_token = saved.get('next_page_token') or None
        if not saved['has_more']:
            print('Previously completed; no provider call.')
            return
    seen = set()
    with api.build_kaggle_client() as client:
        while True:
            response = client.kernels.kernels_api_client.list_kernel_session_output(request)
            for item in response.files or []:
                name = item.file_name
                rows.append(name)
                p = Path(name)
                if p.is_absolute() or '..' in p.parts or p.name not in NAMES or any('.env' in part or 'credential' in part.lower() or part == '.git' for part in p.parts):
                    continue
                raw = urllib.request.urlopen(item.url, timeout=60).read(8 * 1024 * 1024 + 1)
                if len(raw) > 8 * 1024 * 1024:
                    raise RuntimeError('evidence file too large')
                dest = BASE / 'runtime-evidence-current' / p
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists():
                    if dest.read_bytes() != raw:
                        raise RuntimeError('existing evidence differs')
                else:
                    with dest.open('xb') as handle:
                        handle.write(raw)
                retained.append({'path': name, 'sha256': hashlib.sha256(raw).hexdigest()})
            token = response.next_page_token
            progress.write_text(json.dumps({'scope': 'CURRENT_IDENTITY_ONLY', 'files': sorted(set(rows)), 'retained': retained, 'has_more': bool(token), 'next_page_token': token}, indent=2) + '\n')
            if not token:
                break
            if token in seen or len(seen) >= 400:
                raise RuntimeError('paging did not terminate within bounds')
            seen.add(token)
            request.page_token = token
    result = {'target': TARGET, 'scope': 'CURRENT_IDENTITY_ONLY, not exact-version output', 'files': sorted(set(rows)), 'retained': retained}
    (BASE / 'runtime-evidence-retrieval.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'listed_files': len(set(rows)), 'retained': retained, 'scope': result['scope']}, indent=2))

if __name__ == '__main__':
    main()
