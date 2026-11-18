"""Observe one already-pushed Kaggle version, download once, assess locally.

Never pushes, retries a download, or submits to a competition. Run detached only
AFTER the authorized push. Use the unique one-push kernel identity: the installed
Kaggle CLI accepts version suffixes but its status/output API ignores them.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CANDIDATE = ROOT/'deployment/kaggle/q6-prediction-feedback-20260906'
KAGGLE = Path('/home/pranav/.local/share/uv/tools/kaggle/bin/kaggle')
PYTHON = ROOT/'deployment/kaggle/.venv/bin/python'
TERMINAL = {'COMPLETE', 'ERROR', 'CANCELED', 'CANCELLED'}
ACTIVE = {'QUEUED', 'RUNNING', 'INITIALIZING'}


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def command(argv, timeout):
    result = subprocess.run([str(x) for x in argv], capture_output=True, text=True, timeout=timeout)
    return {'argv':[str(x) for x in argv], 'returncode':result.returncode,
            'stdout':result.stdout, 'stderr':result.stderr}


def status_from(text):
    matches = re.findall(r'KernelWorkerStatus\.([A-Z_]+)', text)
    if len(matches) != 1:
        raise ValueError('Kaggle status response did not identify exactly one kernel status')
    status = matches[0]
    if status not in TERMINAL | ACTIVE:
        raise ValueError(f'Unexpected kernel status {status}')
    return status


def snapshot(candidate, destination):
    launch = json.loads((candidate/'LAUNCH.json').read_text())
    for name, expected in launch['files'].items():
        if hashlib.sha256((candidate/name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Candidate changed since freeze: {name}')
    chassis = json.loads((candidate/'CHASSIS.json').read_text())
    if chassis['contract_sha256'] != launch['contract_sha256']:
        raise ValueError('CHASSIS and launch identities differ')
    for name, expected in {**chassis['files'], **chassis['sources']}.items():
        if hashlib.sha256((candidate/name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Candidate source changed since freeze: {name}')
    shutil.copytree(candidate, destination, ignore=shutil.ignore_patterns('__pycache__'))
    return launch


def monitor(kernel, output, *, candidate=CANDIDATE, poll_seconds=60,
            max_wait_seconds=86400, run=command, sleep=time.sleep, clock=time.monotonic):
    output.mkdir(parents=True, exist_ok=True)
    # Exclusive durable marker prevents a second monitor/download for this path,
    # including after a crash. Recovery is an explicit new human decision.
    with (output/'monitor.started.json').open('x') as stream:
        json.dump({'kernel':kernel,'started_at':now(),'pid':os.getpid()}, stream)
    outcome = {'schema':'q6-long-run-monitor-v1','kernel':kernel,'started_at':now(),
               'status':'RUNNING','download_attempts':0,'pushes':0,'competition_submissions':0}
    write(output/'MONITOR.json', outcome)
    try:
        frozen = output/'candidate'
        launch = snapshot(candidate, frozen)
        if kernel not in {launch['kernel']} and not kernel.startswith(launch['kernel']+'/'):
            raise ValueError('Requested kernel differs from candidate launch identity')
        outcome['contract_sha256'] = launch['contract_sha256']
        # Freeze analysis code too so a future edit cannot change this verdict.
        replay = output/'replay_final_feedback.py'
        shutil.copyfile(HERE/'replay_final_feedback.py', replay)
        summarizer = output/'summarize_long_run.py'
        shutil.copyfile(HERE/'summarize_long_run.py', summarizer)
        started = clock()
        while True:
            if clock()-started >= max_wait_seconds:
                raise TimeoutError('Local monitoring deadline expired; kernel status is unresolved')
            response = run([KAGGLE,'kernels','status',kernel], 90)
            with (output/'status.jsonl').open('a') as stream:
                stream.write(json.dumps({'at':now(),**response})+'\n')
            if response['returncode']:
                raise RuntimeError('Kaggle status command failed; see status.jsonl')
            status = status_from(response['stdout'])
            outcome['kernel_status'] = status
            write(output/'MONITOR.json', outcome)
            if status in TERMINAL:
                break
            sleep(min(poll_seconds, max_wait_seconds-(clock()-started)))
        outcome['download_attempts'] = 1
        write(output/'MONITOR.json', outcome)
        raw = output/'raw'
        raw.mkdir()
        response = run([KAGGLE,'kernels','output',kernel,'--path',raw], 3600)
        write(output/'download.json', response)
        if response['returncode']:
            raise RuntimeError('Single terminal-output download failed; no retry attempted')
        evidence = raw/'nooa-join-public3'
        if not evidence.is_dir():
            raise RuntimeError('Downloaded output lacks nooa-join-public3 evidence')
        assessment = run([PYTHON, frozen/'assess.py', evidence, '--output',output/'SCORE.json'], 120)
        write(output/'assessment-command.json', assessment)
        replays = []
        for session in sorted((evidence/'solver-workspaces').glob('*')):
            if not session.is_dir():
                continue
            destination = output/f'model-feedback-final-replay-{session.name}.json'
            result = run([PYTHON,replay,session,'--chassis',frozen/'chassis','--output',destination], 30)
            write(output/f'final-feedback-{session.name}-command.json', result)
            replays.append({'session':session.name,'returncode':result['returncode'],'output':str(destination)})
        outcome['final_replays'] = replays
        report = run([PYTHON,summarizer,evidence,'--output-dir',output], 120)
        write(output/'behavioral-report-command.json', report)
        outcome['behavioral_report_returncode'] = report['returncode']
        if status != 'COMPLETE':
            raise RuntimeError(f'Kaggle ended {status}; downloaded evidence retained for diagnosis')
        if assessment['returncode']:
            raise RuntimeError('Host assessment failed; see SCORE.json and assessment-command.json')
        if any(item['returncode'] for item in replays):
            raise RuntimeError('Final prediction replay failed; host score retained separately')
        if report['returncode']:
            raise RuntimeError('Behavioral report generation failed; raw and host score retained')
        outcome['status'] = 'HOST_PASS_REQUIRES_BEHAVIORAL_REVIEW'
        outcome['claim_ceiling'] = 'Host qualification only; compare raw predictions and actions before judging solver improvement.'
    except Exception as error:
        outcome.update(status='FAILED', error=f'{type(error).__name__}: {error}')
    finally:
        outcome['ended_at'] = now()
        write(output/'MONITOR.json', outcome)
    return outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kernel', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--poll-seconds', type=int, default=60, choices=range(1,61), metavar='1..60')
    parser.add_argument('--max-wait-seconds', type=int, default=86400)
    args = parser.parse_args()
    if args.max_wait_seconds <= 0:
        parser.error('--max-wait-seconds must be positive')
    result = monitor(args.kernel, args.output.resolve(), poll_seconds=args.poll_seconds,
                     max_wait_seconds=args.max_wait_seconds)
    print(json.dumps(result, indent=2), flush=True)
    raise SystemExit(result['status'] != 'HOST_PASS_REQUIRES_BEHAVIORAL_REVIEW')


if __name__ == '__main__':
    main()
