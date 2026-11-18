"""One bounded read-only recovery from preserved successful downloads.

The SDK opens each file before evaluating response.content, so an interrupted
body leaves an empty file; nonempty completed files are reusable. Conflicting
nonempty copies fail closed. Frozen experiment and automatic collector unchanged.
"""
import json
import shutil
import time
from pathlib import Path
import requests
from prepare import HERE, SLUG, digest
from run import REF, api_setup, now, save, verify


def main():
    verify()
    marker = HERE / 'collection-recovery-attempt.json'
    with marker.open('x') as f:
        json.dump({'utc': now(), 'operation': 'read-only missing-file recovery', 'cap_seconds': 900}, f)
    api, _ = api_setup()
    assert str(api.kernels_status(REF).status).endswith('.COMPLETE')
    deadline = time.monotonic() + 900
    raw = json.loads((HERE / 'evidence/output-page1-recovery.json').read_text())
    files = list(raw['files'])
    kernel_log = raw['log']
    token = raw.get('nextPageToken')
    Request = api.kernels_output.__func__.__globals__['ApiListKernelSessionOutputRequest']
    pages = 1
    while token:
        assert pages < 10 and time.monotonic() < deadline
        r = Request()
        r.user_name, r.kernel_slug = REF.split('/')
        api._set_paging(r, 100, token)
        with api.build_kaggle_client() as client:
            response = client.kernels.kernels_api_client.list_kernel_session_output(r)
        raw = json.loads(str(response))
        pages += 1
        save(HERE / f'evidence/output-page{pages}-recovery.json', raw)
        files.extend(raw['files'])
        token = raw.get('nextPageToken')
    names = [x['fileName'] for x in files]
    assert len(names) == len(set(names))
    folder = HERE / 'output.recovery'
    folder.mkdir(exist_ok=False)
    sources = sorted(HERE.glob('output.partial*'))
    provenance = {}
    for item in files:
        assert time.monotonic() < deadline
        name = item['fileName']
        rel = Path(name)
        assert not rel.is_absolute() and '..' not in rel.parts
        dest = folder / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        candidates = [s / rel for s in sources if (s / rel).is_file() and (s / rel).stat().st_size > 0]
        if candidates:
            assert len({digest(p) for p in candidates}) == 1, 'conflicting preserved files: ' + name
            shutil.copy2(candidates[0], dest)
            provenance[name] = {'reused': str(candidates[0].relative_to(HERE)), 'sha256': digest(dest)}
        else:
            response = requests.get(item['url'], timeout=(15, 60))
            response.raise_for_status()
            data = response.content
            if response.headers.get('Content-Length') and not response.headers.get('Content-Encoding'):
                assert len(data) == int(response.headers['Content-Length'])
            dest.write_bytes(data)
            provenance[name] = {'downloaded_utc': now(), 'bytes': len(data), 'sha256': digest(dest)}
            print('recovered', name, len(data), flush=True)
        save(HERE / 'collection-recovery-progress.json', provenance)
    assert kernel_log
    (folder / (SLUG + '.log')).write_text(kernel_log)
    inventory = {str(p.relative_to(folder)): {'bytes': p.stat().st_size, 'sha256': digest(p)} for p in folder.rglob('*') if p.is_file()}
    save(HERE / 'collection.json', {'utc': now(), 'ref': REF, 'version': 1, 'files': inventory,
         'api_file_count': len(files) + 1, 'pages': pages, 'recovery': 'preserved nonempty SDK downloads plus missing files and API kernel log',
         'provenance': 'collection-recovery-progress.json'})
    folder.rename(HERE / 'output')
    print('COMPLETE', len(inventory), flush=True)


if __name__ == '__main__':
    main()
