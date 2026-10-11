"""One-attempt staged execution; resumable collection, no launch retries.

Run with the installed Kaggle CLI Python. Only --run admits new launches.
"""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent / 'marlin-fixed-replay-r3-20261011'
sys.path.insert(0, str(BACKEND))
from kaggle_backend import KaggleBackend

CEILING = 21600
TERMINAL = {'COMPLETE', 'ERROR', 'CANCELED', 'CANCELLED'}


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n'); temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def verify_freeze():
    for name, digest in json.loads((HERE / 'freeze.json').read_text())['files'].items():
        path = Path(name) if Path(name).is_absolute() else HERE / name
        if sha(path) != digest:
            raise ValueError('Frozen source changed: ' + name)


def admission(backend, runs):
    verify_freeze()
    quota = backend.quota().gpu_quota
    used, reserved, allowed = (getattr(quota, name).total_seconds() for name in ['time_used', 'time_reserved', 'total_time_allowed'])
    active = backend.active_sessions('pranavagrawai')
    cap = sum(r['hard_cap_s'] for r in runs)
    all_runs = json.loads((HERE / 'runs.json').read_text())['runs']
    committed = sum(r['hard_cap_s'] for r in all_runs if (HERE / r['run'] / 'launch-attempt.json').exists())
    result = {'utc': now(), 'used_s': used, 'reserved_s': reserved, 'allowed_s': allowed,
              'active': active, 'new_cap_s': cap, 'committed_cap_s': committed}
    if min(used, reserved) < 0 or used + reserved + cap > min(allowed, 108000):
        raise ValueError('Insufficient or inconsistent live quota')
    if len(active) + len(runs) > 2 or any(not a.get('ref') for a in active):
        raise ValueError('Insufficient or unknown GPU slots')
    if committed + cap > CEILING:
        raise ValueError('Experiment six-hour ceiling would be exceeded')
    for run in runs:
        if (HERE / run['run'] / 'launch-attempt.json').exists():
            raise ValueError('Run already attempted; never retry')
        backend.require_new_slug(run['slug'])
    return result


def launch(backend, run, admitted):
    verify_freeze()
    root = HERE / run['run']
    row = {'utc': now(), 'state': 'attempting', 'run': run, 'admission': admitted}
    with (root / 'launch-attempt.json').open('x') as f:
        json.dump(row, f, indent=2)
    response = backend.push(root / 'kernel', run['hard_cap_s'])
    save(root / 'launch-response.json', response)
    if (response.get('error') or response.get('versionNumber') != 1 or not response.get('kernelId')
            or response.get('url') != 'https://www.kaggle.com/code/' + run['slug']
            or any(response.get(key) for key in ['invalidTags', 'invalidDatasetSources', 'invalidCompetitionSources', 'invalidKernelSources', 'invalidModelSources'])):
        raise ValueError('Acceptance unresolved; no retry')
    row.update(state='accepted', response=response)
    save(root / 'launch-attempt.json', row)
    print(now(), 'ACCEPTED', run['slug'], response['kernelId'], flush=True)


def collect(backend, run):
    root = HERE / run['run']
    if (root / 'collection.json').exists():
        receipt = json.loads((root / 'collection.json').read_text())
        for name, digest in receipt['hashes'].items():
            if sha(root / 'output' / name) != digest:
                raise ValueError('Collected output changed: ' + name)
        return receipt
    response = json.loads((root / 'launch-response.json').read_text())
    notebook = json.loads((root / 'kernel' / run['metadata']['code_file']).read_text())
    deadline = time.monotonic() + 12 * 3600
    while time.monotonic() < deadline:
        status = backend.status(run['slug'])
        with (root / 'status-history.jsonl').open('a') as f:
            f.write(json.dumps({'utc': now(), **status}) + '\n')
        if status['status'] in TERMINAL:
            break
        time.sleep(45)
    else:
        raise TimeoutError('Read-only collection deadline; no relaunch')
    remote = backend.verify_remote(run['slug'], response, notebook, run['metadata'])
    remote.pop('source'); save(root / 'remote-source-verification.json', remote)
    output = root / 'output'; output.mkdir(exist_ok=True)
    token = None; seen = set()
    while True:
        files, next_token, log = backend.output_page(run['slug'], token)
        for item in files:
            backend.download(item, output)
        if log:
            (output / 'kernel.log').write_text(log)
        if not next_token:
            break
        if next_token in seen:
            raise ValueError('Repeated output page')
        seen.add(next_token); token = next_token
    backend.verify_remote(run['slug'], response, notebook, run['metadata'])
    hashes = {str(p.relative_to(output)): sha(p) for p in output.rglob('*') if p.is_file()}
    if not hashes:
        raise ValueError('Empty output collection')
    receipt = {'utc': now(), 'status': status['status'], 'kernel_id': response['kernelId'], 'hashes': hashes}
    save(root / 'collection.json', receipt)
    print(now(), 'COLLECTED', run['run'], status['status'], len(hashes), flush=True)
    return receipt


def probe_passed(receipt):
    output = HERE / 'probe/output/pruning'
    if receipt['status'] != 'COMPLETE':
        return False
    target = json.loads((output / 'target.json').read_text())
    draft = json.loads((output / 'draft.json').read_text())
    smoke = json.loads((output / 'SMOKE.json').read_text())
    info = json.loads((output / 'server-info-start.json').read_text())
    states = info.get('internal_states', [])
    if not states or any(s.get('effective_max_running_requests_per_dp', 0) < 18 for s in states):
        return False
    expected = {'kept': 193536, 'dropped': 27648, 'routers': 48}
    return (target['complete'] and target['target_experts'] == 448 and target['counts'] == expected
            and draft['draft_experts'] == 512 and smoke['passed']
            and [w['concurrency'] for w in smoke['windows']] == [14, 18]
            and all(len(w['responses']) == w['concurrency'] for w in smoke['windows']))


def board(message):
    subprocess.run([sys.executable, '/home/pranav/.agent-board/board.py', 'reply', '659', '--as', 'codex-pruning-transfer', '--body', message], check=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--run', action='store_true'); args = parser.parse_args()
    with (HERE / 'execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify_freeze()
        runs = json.loads((HERE / 'runs.json').read_text())['runs']
        backend = KaggleBackend()
        probe, *pair = runs
        if not args.run:
            print(json.dumps(admission(backend, [probe]), indent=2)); return
        if not (HERE / 'probe/launch-attempt.json').exists():
            launch(backend, probe, admission(backend, [probe]))
            board('Pruning transfer probe accepted once. One slot claimed for at most 1 h. If qualification passes, the frozen c14 versus REAP448+c18 pair may claim two slots after fresh admission, 2.5 h each; six-hour total ceiling. No retries or submissions.')
        receipt = collect(backend, probe)
        passed = probe_passed(receipt)
        save(HERE / 'probe/qualification.json', {'passed': passed, 'collection': receipt})
        if not passed:
            result = {'status': 'PROBE_FAILED_RETAIN_CONTROL', 'retained': 'c14', 'gameplay_launched': False}
            save(HERE / 'RESULT.json', result); save(HERE / 'retention.json', result)
            board('Pruning probe failed qualification; c14 retained. No gameplay launches or retry. Evidence: analysis/pruning-transfer-20261011/probe/output and qualification.json.'); return
        pending = [run for run in pair if not (HERE / run['run'] / 'launch-attempt.json').exists()]
        if pending:
            admitted = admission(backend, pending)
            board('Pruning loader and serving qualification passed. Claiming available slots for the frozen c14 control and REAP448+c18 candidate, <=2.5 h each. Score-first retention; no submissions or retries.')
            for run in pending:
                launch(backend, run, admission(backend, [run]))
        for run in pair:
            collect(backend, run)
        # analyze.py uses only stdlib; keep the same estimator as retained c14.
        sys.path.insert(0, str(HERE))
        from evaluate import evaluate
        result = evaluate()
        board('Pruning transfer comparison complete: ' + json.dumps({k: result[k] for k in ['status', 'retained']}) + '. Raw paired results, exact identities and fallback are recorded in analysis/pruning-transfer-20261011/{RESULT.json,retention.json}. No submission.')
        print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        save(HERE / 'execution-stop.json', {'utc': now(), 'error_type': type(error).__name__, 'message': str(error)[:800],
                                           'retained': 'c14', 'retry': False})
        raise
