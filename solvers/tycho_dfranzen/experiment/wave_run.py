"""Wave launch (any subset of frozen arms, at most two GPU sessions at once) and
read-only collection. Reserves every arm at the first launch. No automatic retry."""
from __future__ import annotations
import argparse
import datetime as dt
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pair_run import save, sha, quota  # noqa: E402

HYBRID_ARMS = ('port', 'lean')
TERMINAL = ('COMPLETE', 'ERROR', 'CANCELLED', 'FAILED')


def verify_frozen(directory, identity):
    qualification = json.loads((directory/'LOCAL-QUALIFICATION.json').read_text())
    assert qualification['passed'] and qualification['payload_sha256'] == identity['source_payload_sha256']
    assert qualification['dataset_readback_sha256'] == identity['source_payload_sha256']
    assert (directory/'EXPERIMENT-CARD.md').is_file()
    for arm, item in identity['arms'].items():
        root = directory/arm
        assert sha(root/'kernel/notebook.ipynb') == item['notebook_sha256']
        assert sha(root/'kernel/kernel-metadata.json') == item['metadata_sha256']
        assert json.loads((root/'kernel/kernel-metadata.json').read_text())['is_private'] is True
        if arm in HYBRID_ARMS:
            assert sha(root/'source-bundle/hybrid-runtime.zlib') == identity['source_payload_sha256']


def reserve_all(identity, budget_path):
    """Idempotent: the first launch reserves every arm's full hard cap."""
    with budget_path.with_suffix('.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        budget = json.loads(budget_path.read_text())
        prefix = identity['pair_id']+'-'
        if any(r['run_id'].startswith(prefix) for r in budget['run_reservations']):
            return
        assert budget['unreserved_gpu_hours'] >= identity['reserved_gpu_hours'], 'campaign budget exhausted'
        for arm in identity['launch_order']:
            budget['run_reservations'].append({'run_id': prefix+arm, 'state': 'reserved',
                'reserved_gpu_hours': identity['hard_timeout_seconds']/3600, 'gpu_count': 1,
                'kernel': identity['arms'][arm]['kernel'], 'hard_timeout_seconds': identity['hard_timeout_seconds']})
        budget['unreserved_gpu_hours'] -= identity['reserved_gpu_hours']
        save(budget_path, budget)


def launch(directory, api, budget_path, arms):
    identity = json.loads((directory/'pair-identity.json').read_text())
    assert arms and set(arms) <= set(identity['arms']), arms
    assert len(arms) <= 2, 'Kaggle runs at most two GPU sessions; queued ones count'
    verify_frozen(directory, identity)
    for arm in arms:
        assert not (directory/arm/'launch-attempt.json').exists(), f'{arm} already attempted; do not retry'
    if not (directory/'quota-before.json').exists():
        save(directory/'quota-before.json', quota(api))
    save(directory/f'quota-before-{"-".join(arms)}.json', quota(api))
    reserve_all(identity, budget_path)
    for arm in [a for a in identity['launch_order'] if a in arms]:
        root = directory/arm
        attempt = {'utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'state': 'attempting',
                   'identity': identity['arms'][arm]}
        save(root/'launch-attempt.json', attempt)
        try:
            result = json.loads(str(api.kernels_push(str(root/'kernel'), timeout=str(identity['hard_timeout_seconds']))))
            save(root/'launch-response.json', result)
            if result.get('error') or not result.get('versionNumber'):
                raise RuntimeError('Kaggle did not return an accepted version')
            attempt.update(state='accepted', response=result)
            print(json.dumps({'arm': arm, 'response': result}), flush=True)
        except Exception as error:
            attempt.update(state='acceptance unresolved; inspect, do not retry', error_type=type(error).__name__)
            raise
        finally:
            save(root/'launch-attempt.json', attempt)
            with budget_path.with_suffix('.lock').open('w') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                budget = json.loads(budget_path.read_text())
                row = next(r for r in budget['run_reservations'] if r['run_id'] == identity['pair_id']+'-'+arm)
                row.update(state=attempt['state'], accepted_at=attempt['utc'])
                if 'response' in attempt: row['version'] = attempt['response']['versionNumber']
                save(budget_path, budget)


def collect_arm(directory, arm, kernel, version):
    from collect_outputs import KaggleOutput, safe_path
    remote = KaggleOutput(kernel, version)
    root = directory/arm
    output = root/'output'; output.mkdir(exist_ok=True)
    receipt = {'files': [], 'errors': [], 'critical_complete': False}
    hybrid = arm in HYBRID_ARMS

    def fetch(name, required=True, expected=None):
        try:
            name = safe_path(name)
            data = remote.fetch(name)
            digest = hashlib.sha256(data).hexdigest()
            if expected and digest != expected: raise ValueError('hash mismatch')
            path = output/name; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            receipt['files'].append({'file': name, 'sha256': digest, 'bytes': len(data)})
            return data
        except Exception as error:
            receipt['errors'].append({'file': name, 'required': required, 'error_type': type(error).__name__})
            return None
        finally:
            save(root/'collection-receipt.json', receipt)
    for name in ('qualification-start.json', 'pair-config.json', 'qualification-complete.json'):
        fetch(name)
    for name in ('notebook.log', 'watchdog-receipt.json', 'serve.log'):
        fetch(name, required=False)
    if hybrid: fetch('tycho-dfranzen/evidence-index.json')
    raw = fetch('pair-evidence-index.json')
    if raw:
        for item in json.loads(raw)['files']:
            if item['critical']:
                fetch(item['path'], expected=item['sha256'])
    else:
        fetch('tycho-dfranzen/benchmark.json' if hybrid else 'benchmark.json', required=False)
    receipt['critical_complete'] = not any(e['required'] for e in receipt['errors'])
    save(root/'collection-receipt.json', receipt)


def watch(directory, api, hours=14):
    identity = json.loads((directory/'pair-identity.json').read_text())
    arms = identity['launch_order']
    complete = {arm for arm in arms if (directory/arm/'collection-receipt.json').exists()}
    until = time.monotonic()+hours*3600
    with (directory/'watch.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while len(complete) < len(arms) and time.monotonic() < until:
            for arm in arms:
                root = directory/arm
                if arm in complete or not (root/'launch-response.json').exists(): continue
                try:
                    response = json.loads((root/'launch-response.json').read_text())
                    status = str(api.kernels_status(identity['arms'][arm]['kernel']).status)
                    row = {'utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'status': status}
                    with (root/'status-history.jsonl').open('a') as stream: stream.write(json.dumps(row)+'\n')
                    save(root/'status.json', row)
                    if any(word in status.upper() for word in TERMINAL):
                        collect_arm(directory, arm, identity['arms'][arm]['kernel'], response['versionNumber'])
                        complete.add(arm)
                        save(directory/f'quota-after-{arm}.json', quota(api))
                except Exception as error:
                    save(root/'watch-error.json', {'error_type': type(error).__name__,
                                                   'utc': dt.datetime.now(dt.timezone.utc).isoformat()})
            if len(complete) < len(arms): time.sleep(30)
        save(directory/'quota-after.json', quota(api))
        save(directory/'monitor-result.json', {'collected_arms': sorted(complete),
             'verdict': 'requires receipt, score and mechanism audit; no follow-up launched'})


def main():
    from kaggle.api.kaggle_api_extended import KaggleApi
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['launch', 'watch'])
    parser.add_argument('directory', type=Path)
    parser.add_argument('--arms', nargs='+', default=None, help='launch: subset of arms (max two)')
    args = parser.parse_args()
    api = KaggleApi(); api.authenticate()
    if args.mode == 'launch': launch(args.directory, api, HERE/'budget.json', args.arms)
    else: watch(args.directory, api)


if __name__ == '__main__':
    main()
