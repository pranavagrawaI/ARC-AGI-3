"""Frozen qualification gate; reads completed smoke outputs only."""
import collections
import datetime as dt
import importlib.util
import json
import re
from pathlib import Path
from prepare import HERE, ROOT
from run import save, now, verify


def assess():
    verify()
    out = HERE / 'output'
    issues = []
    status = json.loads((HERE / 'terminal.json').read_text())['status']
    if status.split('.')[-1] != 'COMPLETE':
        issues.append('platform not COMPLETE')
    binding = json.loads((HERE / 'source-readback/identity.json').read_text())
    if not binding['all_cells_equal'] or binding['version'] != 1:
        issues.append('source/version not verified')
    benchmark = json.loads((out / 'benchmark.json').read_text())
    games = benchmark['game_runs']
    expected = json.loads((ROOT / 'solvers/m2_f1x_reap/experiment/screen-1/merge/output/benchmark.json').read_text())
    if len(games) != 25 or {g['game_id'] for g in games} != {g['game_id'] for g in expected['game_runs']}:
        issues.append('game count/identity mismatch')
    duration = (dt.datetime.fromisoformat(benchmark['end_time']) - dt.datetime.fromisoformat(benchmark['start_time'])).total_seconds()
    if not 3600 <= duration <= 3900:
        issues.append('one-hour gameplay horizon not demonstrated')
    if any(g['state'] not in {'won', 'cancelled'} for g in games):
        issues.append('unexpected terminal game state')
    if any(g['solver_note'] and not g['solver_note'].startswith('tokens=') for g in games):
        issues.append('game error-ending note')
    if any(not g['history'] for g in games):
        issues.append('game missing action history')
    kernel = '\n'.join(p.read_text(errors='replace') for p in out.glob('*.log') if p.name != 'serve.log')
    if 'GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition capability: (12, 0)' not in kernel:
        issues.append('actual RTX PRO6000 not demonstrated')
    if 'M2 KEEPALIVE armed: every 360s, hold max 9000s from notebook start' not in kernel or 'M2 KEEPALIVE ready at' not in kernel:
        issues.append('actual keepalive insertion/ready path not demonstrated')
    # Use existing field-tested counters, without the old historical no-harm verdict.
    spec = importlib.util.spec_from_file_location('reap_analyze', ROOT / 'solvers/m2_f1x_reap/experiment/analyze.py')
    analyzer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analyzer)
    serving = analyzer.serving(out)
    arm = analyzer.arm(out)
    target = json.loads((out / 'pruning/target.json').read_text())
    draft = json.loads((out / 'pruning/draft.json').read_text())
    parent_target = json.loads((ROOT / 'solvers/m2_f1x_reap/experiment/screen-1/merge/output/pruning/target.json').read_text())
    checks = {
        'target448': bool(target.get('complete')) and target.get('target_experts') == 448,
        'same_target_mask': target.get('mask_sha256') == parent_target['mask_sha256'],
        'draft512': draft.get('draft_experts') == 512,
        'streams14': serving['gate_14'] and serving['running_max'] == 14,
        'pool_above_f1x': (serving['pool_tokens'] or 0) > 1012736,
        'kv_p90_le_0_90': serving['kv_usage_p90'] <= .90,
        'retractions_le5': serving['retractions'] <= 5,
        'startup_le1300': serving['ready_s'] is not None and serving['ready_s'] <= 1300,
        'preload_zero': arm['preload_name_errors'] == 0,
    }
    issues.extend(key for key, passed in checks.items() if not passed)
    fatal = re.findall(r'OutOfMemoryError|CUDA out of memory|CUDA error:|Watchdog timeout', (out / 'serve.log').read_text(errors='replace'))
    if fatal:
        issues.append('fatal serving errors')
    endpoint_errors = [line[:600] for line in kernel.splitlines() if re.search(r'HTTP Request: POST .*chat/completions.*HTTP/1\.1 (?!200)\d{3}|HTTPError|HTTP Error [45]\d\d|status[_ ]code[=: ]+[45]\d\d', line)]
    per_game = {}
    trace_errors = []
    for game in games:
        path = out / (game['game_id'] + '_p0_requests.jsonl')
        counts = collections.Counter()
        if not path.is_file():
            issues.append('missing trace: ' + game['game_id'])
            continue
        for line_no, line in enumerate(path.open(), 1):
            row = json.loads(line)
            event = row.get('event', 'unknown')
            counts[event] += 1
            if event == 'response':
                counts['completion_tokens'] += int((row.get('usage') or {}).get('completion_tokens') or 0)
            if event in {'error', 'exception', 'request_error'}:
                trace_errors.append({'file': path.name, 'line': line_no, 'event': event, 'error': str(row.get('error'))[:500]})
        if not counts['response']:
            issues.append('no response: ' + game['game_id'])
        per_game[game['game_id']] = {'levels': game['levels_completed'], 'state': game['state'], 'actions': len(game['history']), 'trace': dict(counts)}
    if endpoint_errors or trace_errors:
        issues.append('endpoint/action/trace errors require resolution; qualification blocked')
    levels = sum(g['levels_completed'] for g in games)
    if levels < 67:
        issues.append('no-harm floor67 missed')
    result = {'utc': now(), 'verdict': 'PASS' if not issues else 'FAIL', 'issues': issues, 'levels': levels, 'floor': 67, 'historical_same_parent_levels': [74, 80], 'mean_score': sum(g['final_score'] for g in games) / len(games), 'duration_s': duration, 'checks': checks, 'serving': serving, 'arm': arm, 'per_game': per_game, 'endpoint_errors': endpoint_errors, 'trace_errors': trace_errors, 'fatal_serving_errors': fatal, 'keepalive_lines': [line[:600] for line in kernel.splitlines() if 'M2 KEEPALIVE' in line], 'claim': 'operational/no-harm qualification only; no causal efficacy or promotion'}
    save(HERE / 'QUALIFICATION.json', result)
    print(json.dumps({k: result[k] for k in ['utc', 'verdict', 'issues', 'levels', 'floor', 'serving']}, indent=2))
    return result


if __name__ == '__main__':
    assess()
