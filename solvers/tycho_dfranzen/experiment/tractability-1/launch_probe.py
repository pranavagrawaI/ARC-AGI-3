"""Launch this exact private one-hour freeze once, with a durable reservation."""
from __future__ import annotations

import datetime as dt
import fcntl
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent


def save(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def quota(api):
    with api.build_kaggle_client() as client:
        result = client.kernels.kernels_api_client.get_accelerator_quota_statistics()
    return {'utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'quota': json.loads(str(result))}


def launch():
    from kaggle.api.kaggle_api_extended import KaggleApi
    identity = json.loads((HERE / 'identity.json').read_text())
    checks = {'kernel/notebook.ipynb': 'notebook_sha256', 'kernel/kernel-metadata.json': 'metadata_sha256',
              'source-bundle/hybrid-runtime.zlib': 'source_payload_sha256',
              'source-bundle/branch-packets.zlib': 'packets_payload_sha256', 'EXPERIMENT-CARD.md': 'card_sha256'}
    for name, key in checks.items():
        assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == identity[key], name
    metadata = json.loads((HERE / 'kernel/kernel-metadata.json').read_text())
    assert metadata['is_private'] and metadata['enable_gpu'] and not metadata['enable_internet']
    readback = json.loads((HERE / 'dataset-readback.json').read_text())
    assert readback['passed'] and all(readback[key] == identity[key] for key in
                                    ('source_payload_sha256', 'packets_payload_sha256'))
    qualification = json.loads((HERE / 'LOCAL-QUALIFICATION.json').read_text())
    assert qualification['passed']
    attempt_path = HERE / 'launch-attempt.json'
    api = KaggleApi(); api.authenticate()
    save(HERE / 'quota-before.json', quota(api))
    budget_path = HERE.parent / 'budget.json'
    with budget_path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert not attempt_path.exists(), 'Already attempted; do not retry'
        budget = json.loads(budget_path.read_text())
        assert budget['unreserved_gpu_hours'] >= 1
        assert not any(row['run_id'] == identity['run_id'] for row in budget['run_reservations'])
        budget['run_reservations'].append({'run_id': identity['run_id'], 'state': 'reserved',
            'reserved_gpu_hours': 1, 'gpu_count': 1, 'kernel': identity['kernel'], 'hard_timeout_seconds': 3600})
        budget['unreserved_gpu_hours'] -= 1
        save(budget_path, budget)
        attempt = {'utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'state': 'attempting',
                   'kernel': identity['kernel'], 'identity_sha256': hashlib.sha256((HERE / 'identity.json').read_bytes()).hexdigest(),
                   'hard_timeout_seconds': 3600, 'authority': 'Pranav do it, 2026-10-07; terminal-frame/time-aware plan'}
        save(attempt_path, attempt)
    try:
        response = json.loads(str(api.kernels_push(str(HERE / 'kernel'), timeout='3600')))
        save(HERE / 'launch-response.json', response)
        if response.get('error') or not response.get('versionNumber'):
            raise RuntimeError('No accepted Kaggle version returned; inspect without retry')
        attempt.update(state='accepted', response=response)
        print(json.dumps(response))
    except Exception as error:
        attempt.update(state='acceptance unresolved; do not retry', error_type=type(error).__name__)
        raise
    finally:
        save(attempt_path, attempt)
        with budget_path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            budget = json.loads(budget_path.read_text())
            row = next(row for row in budget['run_reservations'] if row['run_id'] == identity['run_id'])
            row.update(state=attempt['state'], accepted_at=attempt['utc'])
            if 'response' in attempt:
                row['version'] = attempt['response']['versionNumber']
            save(budget_path, budget)


if __name__ == '__main__':
    launch()
