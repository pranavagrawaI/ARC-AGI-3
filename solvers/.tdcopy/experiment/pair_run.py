"""One-shot paired launch and persistent read-only collection. No automatic retry."""
from __future__ import annotations
import argparse
import datetime as dt
import fcntl
import hashlib
import json
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quota(api):
    with api.build_kaggle_client() as client:
        result = client.kernels.kernels_api_client.get_accelerator_quota_statistics()
    return {'utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'quota': json.loads(str(result))}


def launch(directory, api, budget_path):
    identity = json.loads((directory/'pair-identity.json').read_text())
    qualification = json.loads((directory/'LOCAL-QUALIFICATION.json').read_text())
    assert qualification['passed'] and qualification['payload_sha256'] == identity['source_payload_sha256']
    assert (directory/'EXPERIMENT-CARD.md').is_file()
    assert qualification['dataset_readback_sha256'] == identity['source_payload_sha256']
    for arm, item in identity['arms'].items():
        root = directory/arm
        assert sha(root/'kernel/notebook.ipynb') == item['notebook_sha256']
        assert sha(root/'kernel/kernel-metadata.json') == item['metadata_sha256']
        assert json.loads((root/'kernel/kernel-metadata.json').read_text())['is_private'] is True
        assert not (root/'launch-attempt.json').exists(), 'Arm already attempted; do not retry'
    assert sha(directory/'hybrid/source-bundle/hybrid-runtime.zlib') == identity['source_payload_sha256']
    save(directory/'quota-before.json', quota(api))
    # Reserve BOTH arms before the first irreversible API call. Lock the campaign ledger.
    with budget_path.with_suffix('.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        budget = json.loads(budget_path.read_text())
        assert not any(r['run_id'].startswith(identity['pair_id']+'-') for r in budget['run_reservations'])
        assert budget['unreserved_gpu_hours'] >= identity['reserved_gpu_hours']
        for arm in identity['launch_order']:
            item = identity['arms'][arm]
            budget['run_reservations'].append({'run_id': identity['pair_id']+'-'+arm,
                'state': 'reserved', 'reserved_gpu_hours': identity['hard_timeout_seconds']/3600,
                'gpu_count': 1, 'kernel': item['kernel'], 'hard_timeout_seconds': identity['hard_timeout_seconds']})
        budget['unreserved_gpu_hours'] -= identity['reserved_gpu_hours']
        save(budget_path, budget)
    for arm in identity['launch_order']:
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
                row = next(r for r in budget['run_reservations'] if r['run_id']==identity['pair_id']+'-'+arm)
                row.update(state=attempt['state'], accepted_at=attempt['utc'])
                if 'response' in attempt: row['version'] = attempt['response']['versionNumber']
                save(budget_path, budget)


def collect_arm(directory, arm, kernel, version):
    from collect_outputs import KaggleOutput, safe_path
    remote = KaggleOutput(kernel, version)
    root = directory/arm
    output = root/'output'; output.mkdir(exist_ok=True)
    receipt = {'files': [], 'errors': [], 'critical_complete': False}
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
    if arm=='hybrid': fetch('tycho-dfranzen/evidence-index.json')
    raw = fetch('pair-evidence-index.json')
    if raw:
        for item in json.loads(raw)['files']:
            if item['critical']:
                fetch(item['path'], expected=item['sha256'])
    else:
        # Failure outputs may precede the end-cell inventory; retain what exists.
        fetch('tycho-dfranzen/benchmark.json' if arm=='hybrid' else 'benchmark.json', required=False)
    receipt['critical_complete'] = not any(e['required'] for e in receipt['errors'])
    save(root/'collection-receipt.json', receipt)


def watch(directory, api):
    identity = json.loads((directory/'pair-identity.json').read_text())
    complete = set()
    until = time.monotonic()+12*3600
    with (directory/'watch.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        while len(complete)<2 and time.monotonic()<until:
            for arm in identity['launch_order']:
                if arm in complete: continue
                root = directory/arm
                response_path = root/'launch-response.json'
                if not response_path.exists(): continue
                try:
                    response = json.loads(response_path.read_text())
                    status = str(api.kernels_status(identity['arms'][arm]['kernel']).status)
                    row = {'utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'status': status}
                    with (root/'status-history.jsonl').open('a') as stream: stream.write(json.dumps(row)+'\n')
                    save(root/'status.json', row)
                    if any(word in status.upper() for word in ('COMPLETE','ERROR','CANCELLED','FAILED')):
                        collect_arm(directory, arm, identity['arms'][arm]['kernel'], response['versionNumber'])
                        complete.add(arm)
                except Exception as error:
                    save(root/'watch-error.json', {'error_type': type(error).__name__, 'utc': dt.datetime.now(dt.timezone.utc).isoformat()})
            if len(complete)<2: time.sleep(30)
        save(directory/'quota-after.json', quota(api))
        save(directory/'monitor-result.json', {'collected_arms': sorted(complete), 'verdict': 'requires receipt, score and Builder-chain audit; no follow-up launched'})


def main():
    from kaggle.api.kaggle_api_extended import KaggleApi
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['launch', 'watch'])
    parser.add_argument('directory', type=Path)
    args=parser.parse_args()
    api=KaggleApi(); api.authenticate()
    if args.mode=='launch': launch(args.directory, api, HERE/'budget.json')
    else: watch(args.directory, api)


if __name__=='__main__': main()
