"""One-shot continuation: collect existing screen, conditionally run two validations.

No retry, competition submission, publication, input/weight experiment capability.
Run with the installed Kaggle Python; supervised by a user service.
"""
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path

from kaggle import api
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
from policy import BENCH, FLASH, assess_screen, module, validation_decision

HERE = Path(__file__).resolve().parent
KAGGLE = '/home/pranav/.local/bin/kaggle'
NB = 'duck-qwen3-8-flash-next-nvfp4-mtp.ipynb'
SCREEN = 'pranavagrawai/duck-flash-next-speed-screen'


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def save(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(path)


def command(args, log, timeout=60):
    r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    log.write_text(r.stdout + '\n' + r.stderr)
    if r.returncode:
        raise RuntimeError(f'Command failed ({r.returncode}); inspect {log}')
    return r.stdout


def source_check(slug, folder, receipt):
    req = ApiGetKernelRequest()
    req.user_name, req.kernel_slug = slug.split('/')
    with api.build_kaggle_client() as client:
        remote = client.kernels.kernels_api_client.get_kernel(req)
    if remote.metadata.current_version_number != 1:
        raise RuntimeError(f'{slug}: version drift; stop without retry')
    norm = lambda nb: [(c['cell_type'], ''.join(c['source'])) for c in nb['cells']]
    local = json.loads((folder / NB).read_text())
    if norm(local) != norm(json.loads(remote.blob.source)):
        raise RuntimeError(f'{slug}: remote source differs')
    expected = json.loads((folder / 'kernel-metadata.json').read_text())
    for field in ('docker_image', 'machine_shape', 'enable_gpu', 'enable_internet', 'is_private'):
        if getattr(remote.metadata, field) != expected[field]:
            raise RuntimeError(f'{slug}: remote {field} differs')
    save(receipt, {'checked_at': now(), 'slug': slug, 'version': 1, 'source_matches': True,
                   'source_sha256': hashlib.sha256((folder / NB).read_bytes()).hexdigest(),
                   'machine_shape': remote.metadata.machine_shape})


def frozen_check():
    manifest = json.loads((HERE / 'continuation-manifest.json').read_text())
    for file, digest in manifest['files'].items():
        if hashlib.sha256(Path(file).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Frozen continuation source changed: {file}')
    for line in (BENCH / 'execution/manifest.sha256').read_text().splitlines():
        digest, relative = line.split(maxsplit=1)
        if hashlib.sha256((BENCH / relative).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Frozen screen artifact changed: {relative}')


def notify(title, body):
    subprocess.run(['/usr/bin/notify-send', title, body], timeout=10, check=False)


def board(body, suffix):
    command(['/usr/bin/python3', '/home/pranav/.agent-board/board.py', 'reply', '480',
             '--as', 'codex-speed-continuation', '--request-id', 'speed-resume-20260930-' + suffix,
             '--body', body], HERE / ('board-' + suffix + '.log'))


def wait_terminal(slugs, deadlines):
    while True:
        states = {}
        for slug in slugs:
            r = api.kernels_status(slug)
            states[slug] = {'status': r.status.name, 'failure_message': r.failure_message}
        receipt = HERE / 'submission-p/receipt.json'
        if receipt.exists():
            submitted = json.loads(receipt.read_text())
            if submitted['status'] == 'PENDING':
                matches = [r for r in api.competition_submissions('arc-prize-2026-arc-agi-3') if r.ref == submitted['ref']]
                if len(matches) == 1:
                    r = matches[0]
                    submitted.update(status=r.status.name, public_score=r.public_score,
                                     private_score=r.private_score, checked_at=now())
                    save(receipt, submitted)
                    if submitted['status'] != 'PENDING':
                        notify('P submission finished', f"{r.ref}: {r.status.name}, public score {r.public_score}")
        save(HERE / 'watch-status.json', {'updated_at': now(), 'phase': 'monitoring', 'runs': states})
        with (HERE / 'status-history.jsonl').open('a') as f:
            f.write(json.dumps({'at': now(), 'runs': states}) + '\n')
        print(json.dumps(states), flush=True)
        if all(s['status'] not in ('RUNNING', 'QUEUED') for s in states.values()):
            return states
        for slug, s in states.items():
            if s['status'] in ('RUNNING', 'QUEUED') and time.time() >= deadlines[slug]:
                raise RuntimeError(f'{slug}: monitoring deadline exceeded; no new launch. Check remote stop.')
        time.sleep(60)


def download(slug, out):
    out.mkdir(exist_ok=True)
    command([KAGGLE, 'kernels', 'output', slug, '-p', str(out), '-q'],
            HERE / (slug.split('/')[-1] + '-download.log'), timeout=600)


def build_validation(overrides):
    allowed = {'TAAF_VLLM_MTP_INDEX_SHARE_FOR_ITERATION': {'1'},
               'TAAF_VLLM_ENABLE_PREFIX_CACHING': {'1'},
               'TAAF_VLLM_MTP_TOKENS': {'2', '4'},
               'TAAF_VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE': {'60'}}
    if not overrides or any(k not in allowed or v not in allowed[k] for k, v in overrides.items()):
        raise RuntimeError('Unexpected screen overrides; no launch')
    baseline = json.loads((FLASH / 'prune/p1' / NB).read_text())
    for arm in ('pplus1', 'pplus2'):
        folder = HERE / arm
        folder.mkdir(exist_ok=True)
        nb = json.loads(json.dumps(baseline))
        src = ''.join(nb['cells'][3]['source'])
        anchor = 'for key, value in PUBLIC25_VLLM_PROFILE_ENV.items():'
        assert src.count(anchor) == 1
        updated = src.replace(anchor, 'PUBLIC25_VLLM_PROFILE_ENV.update(' + repr(overrides) + ')\n' + anchor)
        nb['cells'][3]['source'] = updated.splitlines(keepends=True)
        assert [(c['cell_type'], c['source']) for i, c in enumerate(nb['cells']) if i != 3] == \
            [(c['cell_type'], c['source']) for i, c in enumerate(baseline['cells']) if i != 3]
        (folder / NB).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + '\n')
        meta = json.loads((FLASH / 'prune/p1/kernel-metadata.json').read_text())
        meta['id'] = 'pranavagrawai/duck-flash-next-' + arm + '-20260930'
        meta['title'] = 'duck-flash-next-' + arm + '-20260930'
        save(folder / 'kernel-metadata.json', meta)
        card = f'''# {arm}-20260930
Question: Does the qualified speed-only P+ retain public-25 performance and raise busy generation throughput?
Posture: execution and validation
Contrast: historical frozen P1/P2 (51/58 levels, busy 384.1/385.4 tok/s) vs two P+ full passes
Prediction: each P+ busy generation median >=422.4 tok/s; pair mean levels >=48.5
Measure: benchmark.json plus all action events; vllm-openai-server.log intervals with running+waiting>=20 during 7920s gameplay; exact remote source and worker flag evidence
Cap: this pass <=10800 seconds / 3 GPU hours, 25 games, 7920 seconds per game; pair <=6 GPU hours; paid API spend not applicable
Stop: one pass, no retry; stop continuation on source drift, failed launch or missing receipts. Kaggle --timeout 10800 enforces the run cap.
Invalid if: missing game/actions, wrong horizon/weights/flags/resource configuration, different game sets, missing final audit, or non-COMPLETE remote status. Existing post-game teardown flag is reported separately.
Result map: both valid, each throughput>=422.4 and pair mean levels>=48.5 -> retain P+ locally; negative/invalid/ambiguous -> retain P. No automatic competition submission or publication.
Outputs: {folder}/out; {HERE}/validation-assessment.json
Selected overrides: {json.dumps(overrides, sort_keys=True)}
Authorization: user's instruction to continue execution of NEXT-PLAN; same frozen speed-only scope.
'''
        (folder / 'EXPERIMENT-CARD.md').write_text(card)
    files = [p for arm in ('pplus1', 'pplus2') for p in (HERE / arm).iterdir() if p.is_file()]
    save(HERE / 'validation-manifest.json', {'frozen_at': now(), 'overrides': overrides,
         'files': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}})


def launch_pair():
    launches = {}
    for arm in ('pplus1', 'pplus2'):
        frozen_check()
        folder = HERE / arm
        manifest = json.loads((HERE / 'validation-manifest.json').read_text())
        for file, digest in manifest['files'].items():
            if hashlib.sha256(Path(file).read_bytes()).hexdigest() != digest:
                raise RuntimeError(f'Frozen validation artifact changed: {file}')
        meta = json.loads((folder / 'kernel-metadata.json').read_text())
        slug = meta['id']
        attempt = folder / 'launch-attempt.json'
        if attempt.exists():
            raise RuntimeError(f'{arm}: launch attempt exists; never repeat an uncertain push')
        started = time.time()
        save(attempt, {'started_at': now(), 'epoch': started, 'slug': slug, 'timeout_s': 10800})
        out = command([KAGGLE, 'kernels', 'push', '-p', str(folder), '--timeout', '10800'],
                      folder / 'push.log', timeout=180)
        if 'Kernel version 1 successfully pushed' not in out:
            raise RuntimeError(f'{arm}: no version-1 acceptance receipt; stop without retry')
        source_check(slug, folder, folder / 'remote-source.json')
        launches[slug] = started + 10800 + 600
        save(HERE / 'validation-launches.json', launches)
    return launches


def collect_validation(states, overrides):
    old = module(HERE.parent / 'arm-p-full-20260930/assess.py', 'p_full_assessor')
    arms = {}
    field_map = {'TAAF_VLLM_MTP_INDEX_SHARE_FOR_ITERATION': ('mtp_index_share_for_iteration', True),
                 'TAAF_VLLM_ENABLE_PREFIX_CACHING': ('enable_prefix_caching', True),
                 'TAAF_VLLM_MTP_TOKENS': ('mtp_speculative_tokens', None),
                 'TAAF_VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE': ('max_cudagraph_capture_size', None)}
    for idx, arm in enumerate(('pplus1', 'pplus2'), 1):
        folder = HERE / arm
        slug = json.loads((folder / 'kernel-metadata.json').read_text())['id']
        source_check(slug, folder, folder / 'remote-source-final.json')
        out = folder / 'out'
        download(slug, out)
        name = 'P+' + str(idx)
        try:
            a = old.summarize(out, name)
            tuning = json.loads((out / 'vllm-server-identity.json').read_text())['vllm_tuning']
            if tuning['kv_cache_memory_bytes'] == 12884901888:
                a['issues'] = [s for s in a['issues'] if s != 'Unexpected KV pool']
            expected = {'kv_cache_memory_bytes': 12884901888, 'max_num_seqs': 12,
                        'max_num_batched_tokens': 8192, 'kv_cache_dtype': 'auto',
                        'mtp_speculative_tokens': 3, 'max_cudagraph_capture_size': 48,
                        'mtp_index_share_for_iteration': False, 'enable_prefix_caching': False}
            for env, value in overrides.items():
                field, boolean = field_map[env]
                expected[field] = boolean if boolean is not None else int(value)
            for field, value in expected.items():
                if tuning.get(field) != value:
                    a['issues'].append(f'Wrong live tuning {field}: {tuning.get(field)} vs {value}')
            text = (out / 'vllm-openai-server.log').read_text(errors='replace')
            args = '\n'.join(re.findall(r'non-default args: (.*)', text))
            if expected['mtp_index_share_for_iteration'] and not re.search(r"index_share_for_mtp_iteration['\"]?\s*:\s*True", args):
                a['issues'].append('Index sharing absent from worker args')
            if expected['enable_prefix_caching']:
                if not re.search(r"enable_prefix_caching['\"]?\s*:\s*True", args) or not re.search(r"[Mm]amba cache mode is set to 'align'", text):
                    a['issues'].append('Prefix caching/align absent from worker log')
        except (OSError, ValueError, KeyError, TypeError) as exc:
            a = {'issues': [str(exc)]}
        if states[slug]['status'] != 'COMPLETE':
            a['issues'].append('Remote status ' + states[slug]['status'])
        arms[name] = a
    decision = validation_decision(arms)
    save(HERE / 'validation-assessment.json', {'arms': arms, 'decision': decision, 'overrides': overrides})
    save(HERE / 'retention.json', {'decided_at': now(), **decision, 'fallback': 'P',
                                 'competition_submission_performed': False, 'overrides': overrides})
    return decision


def main():
    lock = (HERE / 'resume.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    stream = None
    log_file = None
    try:
        frozen_check()
        if (HERE / 'validation-launches.json').exists() or any((HERE / a / 'launch-attempt.json').exists() for a in ('pplus1', 'pplus2')):
            raise RuntimeError('Prior validation launch exists. Manual reconciliation required; no automatic repeat.')
        source_check(SCREEN, BENCH / 'kernel', HERE / 'screen-source.json')
        log_file = (HERE / 'screen-live.log').open('a')
        stream = subprocess.Popen([KAGGLE, 'kernels', 'logs', '-f', SCREEN], stdout=log_file, stderr=subprocess.STDOUT)
        start = dt.datetime.fromisoformat((BENCH / 'execution/pushed_at').read_text().strip().replace('Z', '+00:00')).timestamp()
        states = wait_terminal([SCREEN], {SCREEN: start + 10800})
        stream.terminate()
        stream.wait(timeout=10)
        source_check(SCREEN, BENCH / 'kernel', HERE / 'screen-source-final.json')
        download(SCREEN, HERE / 'screen-out')
        result = json.loads((HERE / 'screen-out/speed-screen.json').read_text())
        outcome = assess_screen(result)
        if states[SCREEN]['status'] != 'COMPLETE':
            outcome['user_plan']['decision'] = {'status': 'INVALID', 'p_plus': None, 'reason': 'Remote screen did not complete'}
        save(HERE / 'screen-assessment.json', outcome)
        d = outcome['user_plan']['decision']
        board('Speed screen collected. User-plan decision: ' + json.dumps(d) + '. Both card and exact-greedy verdicts, raw outputs: ' + str(HERE), 'screen')
        if d['status'] != 'PASS':
            save(HERE / 'retention.json', {'decided_at': now(), 'candidate': 'P', 'screen': d, 'validation_launched': False})
            save(HERE / 'watch-status.json', {'phase': 'complete', 'decision': d, 'candidate': 'P'})
            notify('Speed screen finished', d['status'] + ': keep P. No validation launched.')
            return
        quota = json.loads(command([KAGGLE, 'quota', '--format', 'json'], HERE / 'quota-before-validation.json'))
        remaining = float(next(r['remaining'] for r in quota if r['resource'] == 'GPU').removesuffix('h'))
        if remaining < 6.0:
            raise RuntimeError(f'Only {remaining} GPU hours remain; six-hour validation reserve required. Keep P.')
        build_validation(d['p_plus_overrides'])
        launches = launch_pair()
        board('Qualified P+ validations launched once, version 1 each: ' + json.dumps(list(launches)) + '. Cards and manifests: ' + str(HERE) + '. No submission/publication.', 'validation-start')
        states = wait_terminal(list(launches), launches)
        decision = collect_validation(states, d['p_plus_overrides'])
        command([KAGGLE, 'quota', '--format', 'json'], HERE / 'quota-after.json')
        save(HERE / 'watch-status.json', {'phase': 'complete', 'completed_at': now(), 'decision': decision})
        board('P+ full validation finished: ' + json.dumps(decision) + '. Raw evidence: ' + str(HERE) + '. No new competition submission/publication.', 'result')
        notify('P+ validation finished', json.dumps(decision))
    except Exception as exc:
        save(HERE / 'watch-status.json', {'phase': 'needs_attention', 'at': now(), 'error': str(exc)})
        notify('Speed continuation needs attention', str(exc))
        raise
    finally:
        if stream and stream.poll() is None:
            stream.terminate()
            stream.wait(timeout=10)
        if log_file:
            log_file.close()


if __name__ == '__main__':
    main()
