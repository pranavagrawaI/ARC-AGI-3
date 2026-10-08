"""Read-only status tracking, evidence collection and branch summary; never launch."""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from collect_outputs import KaggleOutput, safe_path


def save(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def collect(directory, remote):
    output = directory / 'output'; output.mkdir(exist_ok=True)
    receipt = {'files': [], 'errors': [], 'critical_complete': False}
    def fetch(name, expected=None, required=False):
        path = output / safe_path(name)
        if path.exists() and expected and hashlib.sha256(path.read_bytes()).hexdigest() == expected:
            return path.read_bytes()
        for attempt in range(4):
            try:
                raw = remote.fetch(name)
                digest = hashlib.sha256(raw).hexdigest()
                if expected and digest != expected: raise ValueError('Output hash mismatch')
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
                receipt['files'].append({'path': name, 'sha256': digest, 'bytes': len(raw)})
                save(directory / 'collection-receipt.json', receipt)
                return raw
            except Exception as error:
                response = getattr(error, 'response', None)
                status = getattr(response, 'status_code', None)
                if status == 429 and attempt < 3:
                    time.sleep(max(60, int(response.headers.get('Retry-After', '60')))); continue
                receipt['errors'].append({'path': name, 'required': required, 'type': type(error).__name__, 'status': status})
                save(directory / 'collection-receipt.json', receipt)
                return None
    for name in ('notebook.log', 'branch-preflight.log', 'qualification-start.json',
                 'qualification-complete.json', 'watchdog-receipt.json', 'serving-readiness.json', 'serve.log'):
        fetch(name)
    raw = fetch('probe-evidence-index.json', required=True)
    if raw:
        for item in json.loads(raw)['files']:
            name = item['path']
            # Authored workspace copies remain remote; collect all requests, raw
            # responses, tool evidence and engine outcomes before optional bulk.
            if '/workspace/' in name: continue
            fetch(name, expected=item['sha256'], required=True)
            time.sleep(0.4)
    else:
        for name in ('tractability-output/suite.json', 'tractability-output/suite-status.json'):
            fetch(name)
    indexed_complete = raw is not None and not any(r['required'] for r in receipt['errors'])
    cases = []
    for path in sorted((output / 'tractability-output').glob('block-*/*/result.json')):
        row = json.loads(path.read_text())
        engine = path.with_name('engine-evaluation.json')
        row['engine'] = json.loads(engine.read_text()) if engine.exists() else None
        row['relative_path'] = str(path.relative_to(output))
        cases.append(row)
    suite_path = output / 'tractability-output/suite-status.json'
    suite_status = json.loads(suite_path.read_text()) if suite_path.exists() else {}
    receipt['critical_complete'] = indexed_complete and len(cases) == 24 and suite_status.get('status') == 'complete'
    save(directory / 'collection-receipt.json', receipt)
    arms = {}
    for row in cases:
        arm = arms.setdefault(row['arm'], {'cases': 0, 'action_commitments': 0, 'verified_level_completions': 0,
                                          'unevaluable_actions': 0, 'unknown_provider_usage_cases': 0})
        arm['cases'] += 1; arm['action_commitments'] += row['action'] is not None
        arm['unknown_provider_usage_cases'] += not row['provider_usage_complete']
        engine = row['engine'] or {}
        arm['verified_level_completions'] += bool(engine.get('prefix_verified') and engine.get('level_completed'))
        arm['unevaluable_actions'] += row['action'] is not None and not engine.get('prefix_verified')
    save(directory / 'summary.json', {'critical_evidence_complete': receipt['critical_complete'], 'arms': arms,
         'cases': cases, 'claim_ceiling': 'Saved-request branch test; no full-game score or general improvement claim. Missing outcomes are not successes.'})


def watch(directory, hours):
    identity = json.loads((directory / 'identity.json').read_text())
    accepted = json.loads((directory / 'launch-response.json').read_text())
    remote = KaggleOutput(identity['kernel'], accepted['versionNumber'])
    until = time.monotonic() + hours * 3600
    with (directory / 'watch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while time.monotonic() < until:
            try:
                status = str(remote.api.kernels_status(identity['kernel']).status)
                row = {'utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'status': status}
                save(directory / 'status.json', row)
                with (directory / 'status-history.jsonl').open('a') as stream: stream.write(json.dumps(row) + '\n')
                if any(word in status.upper() for word in ('COMPLETE', 'ERROR', 'CANCELLED', 'CANCELED', 'FAILED')):
                    collect(directory, remote)
                    with remote.api.build_kaggle_client() as client:
                        quota = client.kernels.kernels_api_client.get_accelerator_quota_statistics()
                    save(directory / 'quota-after.json', {'utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                        'used_gpu_seconds': quota.gpu_quota.time_used.total_seconds(),
                        'allowed_gpu_seconds': quota.gpu_quota.total_time_allowed.total_seconds(),
                        'reserved_gpu_seconds': quota.gpu_quota.time_reserved.total_seconds()})
                    return
            except Exception as error:
                save(directory / 'watch-error.json', {'type': type(error).__name__, 'utc': dt.datetime.now(dt.timezone.utc).isoformat()})
            time.sleep(30)
        save(directory / 'watch-ended.json', {'remote_state': 'unresolved', 'reason': 'Read-only monitor window ended; no relaunch'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--hours', type=float, default=12)
    args = parser.parse_args()
    watch(args.directory.resolve(), args.hours)
