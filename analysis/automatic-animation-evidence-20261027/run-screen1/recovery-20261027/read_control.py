"""Read recovered artifacts only; imports no launch entry point."""
import collections
import datetime as dt
import hashlib
import importlib.util
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
EXP = RUN.parent
ROOT = EXP.parents[1]
OUT = RUN / 'control' / 'output'
PARENT = ROOT / 'solvers/m2_f1x_reap/experiment/screen-1/merge/output/benchmark.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    with (HERE / name).open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')


def summarize(b):
    games = b['game_runs']
    return {
        'games': len(games),
        'unique_games': len({g['game_id'] for g in games}),
        'levels': sum(g['levels_completed'] for g in games),
        'mean_score': sum(g['final_score'] for g in games) / len(games),
        'states': dict(collections.Counter(g['state'] for g in games)),
        'actions': sum(len(g['history']) for g in games),
        'recorded_action_tokens': sum(h.get('generated_tokens', 0) or 0
                                     for g in games for h in g['history']),
        'token_note_games': sum(bool(re.fullmatch(r'tokens=\d+', g.get('solver_note') or ''))
                                for g in games),
        'token_note_total': sum(int(m.group(1)) for g in games
                                if (m := re.fullmatch(r'tokens=(\d+)', g.get('solver_note') or ''))),
        'start_time': b['start_time'],
        'end_time': b['end_time'],
        'duration_s': (dt.datetime.fromisoformat(b['end_time']) -
                       dt.datetime.fromisoformat(b['start_time'])).total_seconds(),
        'wins': [g['game_id'] for g in games if g['state'] == 'won'],
        'error_ending_notes': [{'game': g['game_id'], 'note': g['solver_note']}
                               for g in games if g.get('solver_note') and
                               not re.fullmatch(r'tokens=\d+', g['solver_note'])],
    }


def main():
    collection = json.loads((RUN / 'control/collection.json').read_text())
    assert collection['complete_download'] and OUT.is_dir()
    mismatches = [name for name, record in collection['files'].items()
                  if not (OUT / name).is_file() or sha(OUT / name) != record['sha256']]
    assert not mismatches, mismatches
    binding = json.loads((HERE / 'source-readback/identity.json').read_text())
    assert binding['id'] == 141075118 and binding['current_version_number'] == 1
    assert binding['source_equal']
    spec = importlib.util.spec_from_file_location('frozen_pair_reader', EXP / 'run_pair.py')
    pair = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pair)
    assert pair.verify_sources() == 'edc960c92e07b582384279fd168ccabb3a467cf5f3fedeb2f718a51250fef453'
    old, new = json.loads(PARENT.read_text()), json.loads((OUT / 'benchmark.json').read_text())
    old_games = {g['game_id']: g for g in old['game_runs']}
    assert set(old_games) == {g['game_id'] for g in new['game_runs']}
    health = pair.assess_output(OUT, 'KernelWorkerStatus.COMPLETE', set(old_games))
    save('HEALTH.json', health)
    requests = pair.request_receipts(OUT)
    assert requests['totals'].get('automatic_image_request_exposures', 0) == 0
    games = []
    for g in new['game_runs']:
        before = old_games[g['game_id']]
        games.append({'game': g['game_id'], 'historical_levels': before['levels_completed'],
                      'control_levels': g['levels_completed'],
                      'level_delta': g['levels_completed'] - before['levels_completed'],
                      'historical_score': before['final_score'], 'control_score': g['final_score'],
                      'control_state': g['state']})
    patterns = {
        'gpu': r'GPU: NVIDIA.*',
        'readiness': r'SERVER READY.*',
        'fatal': r'OutOfMemoryError|CUDA out of memory|CUDA error:|Watchdog timeout',
        'triton_warning': r'kernel loading failed',
        'endpoint_error': r'HTTP/[0-9.]+["\s]+[45]\d\d|ConnectionError|Connection refused|ReadTimeout',
    }
    evidence = {k: [] for k in patterns}
    for p in sorted(OUT.glob('*.log')):
        for line_no, line in enumerate(p.read_text(errors='replace').splitlines(), 1):
            for name, pattern in patterns.items():
                if re.search(pattern, line, re.I):
                    evidence[name].append({'file': p.name, 'line': line_no, 'text': line[:1200]})
    summary = {
        'utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'authority': 'User authorized recovery and inspection of completed control; no new GPU run.',
        'classification': 'RECOVERED_CONTROL_HEALTHY' if health['valid'] else 'RECOVERED_CONTROL_UNHEALTHY',
        'collection': {'files': len(collection['files']),
                       'bytes': sum(x['bytes'] for x in collection['files'].values()),
                       'all_hashes_recomputed': True, 'hash_mismatches': mismatches},
        'source_identity': binding,
        'health': health,
        'historical_d16': summarize(old), 'recovered_control': summarize(new),
        'request_totals': requests['totals'],
        'level_changes': dict(collections.Counter('higher' if g['level_delta'] > 0 else
                                                 'lower' if g['level_delta'] < 0 else 'same'
                                                 for g in games)),
        'games': games, 'log_evidence': evidence,
        'limits': ['Historical comparison uses the same notebook and game IDs, but unmatched trajectories and different workers/dates.',
                   'Treatment was never launched; animation benefit and paired no-harm threshold are untested.',
                   'GPU usage remains unknown; wall time and benchmark duration are not GPU accounting.',
                   'Original STOPPED state and closure are preserved; this is a separately authorized recovery.'],
        'artifacts': {'benchmark': str(OUT / 'benchmark.json'), 'parent_benchmark': str(PARENT),
                      'collection': str(RUN / 'control/collection.json'),
                      'request_receipts': str(RUN / 'control/request-receipts.json')},
    }
    save('READOUT.json', summary)
    print(json.dumps({k: summary[k] for k in ['classification', 'collection', 'health',
                                             'recovered_control', 'request_totals', 'level_changes']}))


if __name__ == '__main__':
    main()
