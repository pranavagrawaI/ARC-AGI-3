"""One irreversible push; restartable read-only observation and collection."""
import datetime as dt
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from prepare import HERE, ROOT, SLUG, digest, expected

CAP = 6000
QUEUE_CAP = 18000
OBSERVATION_CAP = 7800
TERMINAL = {'COMPLETE', 'ERROR', 'CANCELLED', 'CANCELED', 'CANCEL_ACKNOWLEDGED'}
REF = 'pranavagrawai/' + SLUG
KPY = '/home/pranav/.local/share/uv/tools/kaggle/bin/python'


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def verify():
    frozen = json.loads((HERE / 'FROZEN.json').read_text())
    for relative, sha in frozen['sha256'].items():
        assert digest(ROOT / relative) == sha, 'frozen drift: ' + relative
    assert json.loads((HERE / 'kernel' / (SLUG + '.ipynb')).read_text()) == expected()
    return digest(HERE / 'FROZEN.json')


def api_setup():
    import requests
    send = requests.Session.send
    def bounded(session, request, **kwargs):
        kwargs.setdefault('timeout', (15, 45))
        return send(session, request, **kwargs)
    requests.Session.send = bounded
    sys.path.insert(0, str(ROOT / 'analysis/mtp-depth-20261006'))
    import launch_watch as lw
    lw.EVID = HERE / 'evidence'
    lw.NOTEBOOK_DIR = HERE / 'kernel'
    lw.refresh_token()
    api = lw.KaggleApi()
    api.authenticate()
    return api, lw


def preflight(api, lw):
    binding = verify()
    assert dt.datetime.now(dt.timezone.utc).date() == dt.date(2026, 10, 30), 'today-only authority'
    active = lw.active_sessions(api)
    used, quota = lw.quota_used_s(api)
    result = {'utc': now(), 'active': active, 'used_plus_reserved_s': used, 'quota': quota, 'binding': binding}
    save(HERE / 'evidence/quota-before.json', result)
    assert not active and quota['gpuQuotaSeconds']['timeReserved'] == 0, 'exclusive GPU admission refused'
    assert used + CAP <= 108000, 'weekly GPU cap'
    limits = subprocess.run(['/home/pranav/.local/bin/kaggle', 'competitions', 'submission-limits', 'arc-prize-2026-arc-agi-3'], capture_output=True, text=True, timeout=60, check=True).stdout
    (HERE / 'evidence/limits-before.txt').write_text(limits)
    assert 'Remaining today: 1' in limits, 'today slot unavailable'
    try:
        api.kernels_status(REF)
    except ValueError as error:
        assert "Permission 'kernels.get' was denied" in str(error), 'unexpected slug check failure'
    else:
        raise RuntimeError('candidate slug already exists; do not create another version')
    return result


def launch():
    marker = HERE / 'launch-attempt.json'
    assert not marker.exists(), 'launch already attempted; never retry'
    assert (HERE / 'evidence/gpu-owner-release.json').exists(), 'owner release receipt required'
    api, lw = api_setup()
    before = preflight(api, lw)
    attempt = {'utc': now(), 'state': 'attempting', 'cap_s': CAP, 'preflight': before}
    with marker.open('x') as f:
        json.dump(attempt, f, indent=2)
    import requests
    original = requests.Session.send
    exchanges = []
    def record(session, request, **kwargs):
        response = original(session, request, **kwargs)
        if request.url.split('?')[0].endswith('/SaveKernel'):
            exchanges.append({'utc': now(), 'status': response.status_code, 'bytes_sent': len(request.body or b''), 'content_type': response.headers.get('Content-Type'), 'response_body': response.text[:2000]})
        return response
    requests.Session.send = record
    try:
        response = json.loads(str(api.kernels_push(str(HERE / 'kernel'), timeout=str(CAP))))
        save(HERE / 'launch-response.json', response)
        assert not response.get('error') and response.get('versionNumber') == 1, 'unexpected push response'
        assert response['ref'].split('/code/')[-1] == REF, 'unexpected canonical identity'
        attempt.update(state='accepted', response=response)
    except BaseException as error:
        attempt.update(state='acceptance unresolved; never retry', error_type=type(error).__name__, error=str(error)[:600])
        raise
    finally:
        requests.Session.send = original
        save(marker, attempt)
        save(HERE / 'evidence/push-http.json', exchanges)
    print(json.dumps(attempt), flush=True)


def source_readback(api, label):
    folder = HERE / label
    folder.mkdir(exist_ok=False)
    api.kernels_pull(REF, str(folder), metadata=True, quiet=True)
    paths = list(folder.glob('*.ipynb'))
    assert len(paths) == 1, 'ambiguous remote source'
    remote = json.loads(paths[0].read_text())
    projection = lambda nb: [(c['cell_type'], ''.join(c['source'])) for c in nb['cells']]
    assert projection(remote) == projection(expected()), 'remote source changed'
    mine = api.kernels_list(mine=True, user='pranavagrawai', sort_by='dateRun', page_size=30)
    record = next(k for k in mine if getattr(k, 'ref', None) == REF)
    version = getattr(record, 'current_version_number', None)
    assert version == 1, 'latest remote version changed'
    result = {'utc': now(), 'ref': REF, 'version': version, 'all_cells_equal': True, 'sha256': digest(paths[0]), 'status': str(api.kernels_status(REF).status)}
    save(folder / 'identity.json', result)
    return result


def observation_bounds(rows, accepted_utc):
    accepted = dt.datetime.fromisoformat(accepted_utc).timestamp()
    started = next((dt.datetime.fromisoformat(r['utc']).timestamp() for r in rows if r.get('status') and r['status'].split('.')[-1] != 'QUEUED'), None)
    return (accepted + QUEUE_CAP) if started is None else (started + OBSERVATION_CAP)


def observe():
    attempt = json.loads((HERE / 'launch-attempt.json').read_text())
    assert attempt['state'] == 'accepted', 'acceptance unresolved; reconcile read-only, never launch'
    api, lw = api_setup()
    if not (HERE / 'source-readback/identity.json').exists():
        try:
            source_readback(api, 'source-readback')
        except Exception as error:
            save(HERE / 'source-readback-error.json', {'utc': now(), 'error_type': type(error).__name__, 'reason': str(error)[:600], 'qualification_blocked_until_verified': True})
    history = HERE / 'status-history.jsonl'
    while True:
        rows = [json.loads(line) for line in history.read_text().splitlines()] if history.exists() else []
        deadline = observation_bounds(rows, attempt['utc'])
        if time.time() > deadline:
            save(HERE / 'observation-paused.json', {'utc': now(), 'reason': 'separate observation bound exceeded', 'accepted_run_unresolved': True, 'never_retry_push': True})
            return
        try:
            status = api.kernels_status(REF)
            row = {'utc': now(), 'status': str(status.status), 'failure': getattr(status, 'failure_message', None)}
        except Exception as error:
            row = {'utc': now(), 'status': None, 'error_type': type(error).__name__}
        with history.open('a') as f:
            f.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
        if row['status'] and row['status'].split('.')[-1] in TERMINAL:
            save(HERE / 'terminal.json', row)
            used, quota = lw.quota_used_s(api)
            save(HERE / 'evidence/quota-terminal.json', {'utc': now(), 'used_plus_reserved_s': used, 'quota': quota})
            return
        time.sleep(60)


def collect():
    assert json.loads((HERE / 'terminal.json').read_text())['status'].split('.')[-1] in TERMINAL
    api, _ = api_setup()
    import requests
    get = requests.get
    def checked_get(*args, **kwargs):
        kwargs.setdefault('timeout', (15, 60))
        response = get(*args, **kwargs)
        response.raise_for_status()
        return response
    requests.get = checked_get
    folder = HERE / 'output.partial'
    folder.mkdir(exist_ok=False)
    files, token = api.kernels_output(REF, path=str(folder), force=False, quiet=True, page_size=100)
    assert not token, 'incomplete output pagination'
    inventory = {str(p.relative_to(folder)): {'bytes': p.stat().st_size, 'sha256': digest(p)} for p in folder.rglob('*') if p.is_file()}
    assert inventory, 'empty collection'
    save(HERE / 'collection.json', {'utc': now(), 'ref': REF, 'version': 1, 'files': inventory, 'api_file_count': len(files)})
    folder.rename(HERE / 'output')


if __name__ == '__main__':
    {'launch': launch, 'observe': observe, 'collect': collect, 'readback': lambda: print(source_readback(api_setup()[0], 'final-source-readback'))}[sys.argv[1]]()
