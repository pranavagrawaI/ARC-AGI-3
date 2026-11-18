"""One-shot P1/P2 collector and readout. No launch, retry, or submission capability.

Runs under the installed Kaggle Python environment. Checks status every 60 s,
downloads final outputs, verifies the remote source, then runs the offline gate.
"""
import datetime as dt
import fcntl
import hashlib
import json
import subprocess
import time
from pathlib import Path

from kaggle import api
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/pranav/projects/arc-agi-3-reference/duck-qwen38/runs/flash-next')
EVIDENCE = ROOT / 'prune/execution-20260930'
ARMS = ['p1', 'p2']


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def save(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)


def command(args, log, timeout=60):
    r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    log.write_text(r.stdout + '\n' + r.stderr)
    if r.returncode:
        raise RuntimeError(f'{args[0]} failed ({r.returncode}); see {log}')
    return r.stdout


def source_check(arm):
    req = ApiGetKernelRequest()
    req.user_name = 'pranavagrawai'
    req.kernel_slug = 'duck-flash-next-' + arm
    with api.build_kaggle_client() as client:
        response = client.kernels.kernels_api_client.get_kernel(req)
    if response.metadata.current_version_number != 1:
        raise RuntimeError(f'{arm}: latest remote version changed; do not mix runs')
    original_path = ROOT / 'prune' / arm / 'duck-qwen3-8-flash-next-nvfp4-mtp.ipynb'
    manifest = json.loads((EVIDENCE / 'manifest.json').read_text())
    if hashlib.sha256(original_path.read_bytes()).hexdigest() != manifest['files'][str(original_path)]:
        raise RuntimeError(f'{arm}: local frozen source changed')
    original = json.loads(original_path.read_text())
    remote = json.loads(response.blob.source)
    normalize = lambda nb: [(c['cell_type'], ''.join(c['source'])) for c in nb['cells']]
    if normalize(original) != normalize(remote):
        raise RuntimeError(f'{arm}: remote notebook source differs')
    expected = json.loads((original_path.parent / 'kernel-metadata.json').read_text())
    meta = response.metadata
    for field in ['docker_image', 'machine_shape', 'enable_gpu', 'enable_internet', 'is_private']:
        if getattr(meta, field) != expected[field]:
            raise RuntimeError(f'{arm}: remote {field} changed')
    receipt = {'checked_at': now(), 'version': 1, 'kernel_id': meta.id,
               'source_matches': True, 'machine_shape': meta.machine_shape,
               'source_sha256': hashlib.sha256(original_path.read_bytes()).hexdigest()}
    save(EVIDENCE / (arm + '-source-verified.json'), receipt)
    return receipt


def main():
    lock = (HERE / 'watch.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {'started_at': now(), 'phase': 'monitoring', 'arms': {}}
    stop_at = max(dt.datetime.fromisoformat(json.loads((EVIDENCE / (a + '-launch.json')).read_text())['started_at']).timestamp() for a in ARMS) + 3 * 3600
    try:
        for arm in ARMS:
            source_check(arm)
        while True:
            terminal = True
            for arm in ARMS:
                response = api.kernels_status('pranavagrawai/duck-flash-next-' + arm)
                status = response.status.name
                state['arms'][arm] = {'status': status, 'failure_message': response.failure_message, 'checked_at': now()}
                terminal = terminal and status not in ['RUNNING', 'QUEUED']
            state['updated_at'] = now()
            save(HERE / 'watch-status.json', state)
            with (HERE / 'status-history.jsonl').open('a') as f:
                f.write(json.dumps(state) + '\n')
            print(json.dumps(state), flush=True)
            if terminal:
                break
            if time.time() >= stop_at:
                raise RuntimeError('Three-hour operational limit reached. Manual Kaggle stop required: installed status API does not expose the session ID required by its cancel API. No new run will be launched.')
            time.sleep(60)
        state['phase'] = 'collecting'
        save(HERE / 'watch-status.json', state)
        for arm in ARMS:
            source_check(arm)
            out = ROOT / ('out-' + arm)
            command(['/home/pranav/.local/bin/kaggle', 'kernels', 'output', 'pranavagrawai/duck-flash-next-' + arm,
                     '-p', str(out), '-q'], EVIDENCE / (arm + '-download.log'), timeout=600)
        command(['/usr/bin/python3', str(HERE / 'assess.py'), '--terminal'], HERE / 'assessment-command.log', timeout=180)
        result = json.loads((HERE / 'assessment.json').read_text())
        if any(v['status'] != 'COMPLETE' for v in state['arms'].values()):
            result['decision'] = {'status': 'INVALID', 'candidate': 'KF', 'reason': 'Remote run did not complete successfully.', 'remote_status': state['arms']}
            save(HERE / 'assessment.json', result)
            # Use the system Python (which has matplotlib) for report regeneration.
            command(['/usr/bin/python3', '-c', 'import json; from pathlib import Path; from assess import render, write_report; p=Path(__import__("sys").argv[1]); r=json.loads((p/"assessment.json").read_text()); render(r,p); write_report(r,p)', str(HERE)], HERE / 'report-command.log', timeout=180)
        decision = result['decision']
        retention = {'decided_at': now(), **decision, 'fallback': 'KF', 'competition_submission_performed': False,
                     'assessment': str(HERE / 'assessment.json'),
                     'candidate_sources': [str(ROOT / 'prune' / a) for a in ARMS] if decision['candidate'] == 'P' else [str(ROOT / 'kf2/kernel')],
                     'frozen_manifest': str(EVIDENCE / 'manifest.json')}
        save(HERE / 'retention.json', retention)
        command(['/home/pranav/.local/bin/kaggle', 'quota', '--format', 'json'], EVIDENCE / 'quota-after.json')
        state.update(phase='complete', completed_at=now(), decision=decision)
        save(HERE / 'watch-status.json', state)
        body = f"P1/P2 completion readout: {json.dumps(decision)}. Raw outputs: {ROOT}/out-p1 and out-p2. Report: {HERE}/REPORT.md; retention.json records the local candidate. No rerun or competition submission performed."
        command(['/usr/bin/python3', '/home/pranav/.agent-board/board.py', 'reply', '480', '--as', 'codex-p-execution', '--request-id', 'codex-p-execution-20260930-result', '--body', body], HERE / 'board-result.log')
        print(json.dumps(retention), flush=True)
        subprocess.run(['/usr/bin/notify-send', 'Arm P full passes finished', f"{decision['status']}: candidate {decision['candidate']}. Read {HERE / 'REPORT.md'}"], timeout=10, check=False)
    except Exception as exc:
        state.update(phase='needs_attention', updated_at=now(), error=str(exc))
        save(HERE / 'watch-status.json', state)
        print(json.dumps(state), flush=True)
        subprocess.run(['/usr/bin/notify-send', '-u', 'critical', 'Arm P needs attention', str(exc)], timeout=10, check=False)
        raise


if __name__ == '__main__':
    main()
