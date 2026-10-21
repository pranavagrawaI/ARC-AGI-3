"""One-attempt launcher; refuses an occupied RTX window or insufficient quota."""
import datetime as dt
import hashlib
import json
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

HERE = Path(__file__).resolve().parent


def save(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2, default=str) + '\n')


def main():
    attempt = HERE / 'launch-attempt.json'
    if attempt.exists():
        raise SystemExit('One launch attempt already recorded. No automatic retry.')
    freeze = json.loads((HERE / 'freeze.json').read_text())
    assert all(hashlib.sha256((HERE / k).read_bytes()).hexdigest() == v for k, v in freeze['files'].items())
    assert json.loads((HERE / 'local-checks.json').read_text())['passed']
    api = KaggleApi(); api.authenticate()
    with api.build_kaggle_client() as client:
        q = client.kernels.kernels_api_client.get_accelerator_quota_statistics()
    used, reserved = q.gpu_quota.time_used.total_seconds(), q.gpu_quota.time_reserved.total_seconds()
    save('quota-prelaunch.json', {'utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                                'used_s': used, 'reserved_s': reserved, 'raw': json.loads(str(q))})
    # The UI's 30 h cap is the established contract; API totalTimeAllowed is unreliable.
    assert used + reserved + 5400 <= 30 * 3600, 'insufficient remaining weekly quota'
    assert reserved == 0, 'another session reserves GPU quota; do not risk a silent T4 fallback'
    rows = []
    for k in api.kernels_list(mine=True, user='pranavagrawai', sort_by='dateRun', page_size=20):
        ref = getattr(k, 'ref', None)
        if not ref:
            if getattr(k, 'title', None) == '[Private Notebook]' and not getattr(k, 'author', None):
                continue
            raise RuntimeError('unreadable kernel identity')
        state = str(api.kernels_status(ref).status)
        rows.append({'ref': ref, 'status': state})
    save('sessions-prelaunch.json', rows)
    assert all('RUNNING' not in r['status'] and 'QUEUED' not in r['status'] for r in rows), 'active notebook exists'
    receipt = {'utc': dt.datetime.now(dt.timezone.utc).isoformat(),
               'kernel': json.loads((HERE/'kernel/kernel-metadata.json').read_text())['id'],
               'hard_timeout_s': 5400, 'freeze_sha256': hashlib.sha256((HERE/'freeze.json').read_bytes()).hexdigest()}
    with attempt.open('x') as f: json.dump(receipt, f, indent=2)
    try:
        response = api.kernels_push(str(HERE/'kernel'), timeout='5400', acc='NvidiaRtxPro6000')
        save('launch-response.json', json.loads(str(response)))
        print(str(response))
    except Exception as e:
        save('launch-error.json', {'error': repr(e), 'no_retry': True})
        raise


if __name__ == '__main__':
    main()
