"""Apply the B1 cards to collected outputs. Invalidity first, then the primary measure, then counters.

  python3 -B analyze_b1.py smoke OUT
  python3 -B analyze_b1.py pair R1_OUT R2_OUT

References for the pair (same notebook settings, full public-25): controls c1 and pc; m2_evidence pt.
"""
import collections, glob, json, re, statistics, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
REFS = {'c1': REPO / 'analysis/m2-winners-20261001/runs/out-control',
        'pc': REPO / 'solvers/m2_evidence/experiment/pair-1/control/output',
        'pt': REPO / 'solvers/m2_evidence/experiment/pair-1/treatment/output'}
CONTROLS = ['c1', 'pc']
EXC = re.compile(r'^(\w+(?:Error|Exception|Interrupt)):', re.M)
MARKERS = {'no_tool_call': 'produced no tool call', 'no_user_query': 'No user query', 'overflow': 'context_overflow_recovered',
           'tool_timeout': 'Tool timed out after', 'import_blocked': 'is not allowed in the sandbox'}


def runs(out):
    b = json.loads((Path(out) / 'benchmark.json').read_text())
    return {g['game_id'][:4]: g for g in b['game_runs']}


def log_text(out):
    logs = sorted(Path(out).glob('*.log'))
    return logs[0].read_text(errors='replace') if logs else ''


def transcript_counts(out):
    exc, marks = collections.Counter(), collections.Counter()
    for f in glob.glob(str(Path(out) / 'transcripts/*.txt')):
        t = Path(f).read_text(errors='replace')
        exc.update(EXC.findall(t))
        for k, s in MARKERS.items():
            marks[k] += t.count(s)
    return dict(exc.most_common(12)), dict(marks)


def b1_counters(out):
    res = {}
    for f in glob.glob(str(Path(out) / 'artifacts/**/*_b1.json'), recursive=True):
        res[Path(f).name[:4]] = json.loads(Path(f).read_text())
    return res


def invalid(out, expect_cfg):
    p, text = [], log_text(out)
    if 'M2 B1 installed' not in text:
        p.append('log lacks "M2 B1 installed"')
    for s in expect_cfg:
        if s not in text:
            p.append(f'log lacks {s!r}')
    return p


def smoke(out):
    r = runs(out)
    report = {'invalid': invalid(out, []) + ([] if len(r) == 3 else [f'{len(r)} games, not 3']),
              'levels': {g: x['levels_completed'] for g, x in r.items()},
              'b1': b1_counters(out)}
    report['exceptions'], report['markers'] = transcript_counts(out)
    c = report['b1']
    report['checks'] = {'counters_all_games': len(c) == 3,
                        'pin_fired_2_games': sum(1 for v in c.values() if v.get('trims_with_pin', 0) >= 1) >= 2,
                        'no_fallbacks': all(v.get('fallbacks', 0) == 0 for v in c.values()),
                        'timeout_60': bool(c) and all(v.get('python_timeout_s') == 60 for v in c.values()),
                        'static_rules_on': bool(c) and all(v.get('static_turn_rules') for v in c.values()),
                        'no_no_user_query': report['markers'].get('no_user_query', 0) == 0}
    print(json.dumps(report, indent=1))


def pair(r1, r2):
    cfg = ['max_runtime_s_per_game=7254', 'concurrency=25', 'priority gate active: 10 concurrent streams']
    arms = {'b1-r1': runs(r1), 'b1-r2': runs(r2), **{k: runs(v) for k, v in REFS.items()}}
    problems = invalid(r1, cfg) + invalid(r2, cfg) + [f'{k} has {len(v)} games' for k, v in arms.items() if len(v) != 25]
    games = sorted(set.intersection(*(set(v) for v in arms.values())))
    lev = {k: sum(v[g]['levels_completed'] for g in games) for k, v in arms.items()}
    sco = {k: statistics.mean(v[g]['final_score'] for g in games) for k, v in arms.items()}
    b1_l = statistics.mean([lev['b1-r1'], lev['b1-r2']]); ctl_l = statistics.mean(lev[c] for c in CONTROLS)
    b1_s = statistics.mean([sco['b1-r1'], sco['b1-r2']]); ctl_s = statistics.mean(sco[c] for c in CONTROLS)
    d_l, d_s = b1_l - ctl_l, b1_s - ctl_s
    if problems:
        verdict = 'INVALID'
    elif d_l >= 15 and d_s >= 8:
        verdict = 'POSITIVE'
    elif d_l <= -10:
        verdict = 'NEGATIVE'
    else:
        verdict = 'INCONCLUSIVE'
    per_game = {}
    for g in games:
        b = statistics.mean([arms['b1-r1'][g]['levels_completed'], arms['b1-r2'][g]['levels_completed']])
        c = statistics.mean(arms[k][g]['levels_completed'] for k in CONTROLS)
        per_game[g] = {'b1': [arms['b1-r1'][g]['levels_completed'], arms['b1-r2'][g]['levels_completed']],
                       'controls': [arms[k][g]['levels_completed'] for k in CONTROLS], 'm2_evidence': arms['pt'][g]['levels_completed'],
                       'delta': round(b - c, 1)}
    counters = {k: b1_counters(v) for k, v in (('b1-r1', r1), ('b1-r2', r2))}
    tx = {k: transcript_counts(v) for k, v in (('b1-r1', r1), ('b1-r2', r2), ('pt', REFS['pt']), ('pc', REFS['pc']))}
    report = {'verdict': verdict, 'invalid_reasons': problems,
              'levels': {**lev, 'b1_mean': b1_l, 'control_mean': ctl_l, 'delta_vs_control': d_l, 'delta_vs_m2_evidence': b1_l - lev['pt']},
              'mean_score': {**{k: round(v, 2) for k, v in sco.items()}, 'delta_vs_control': round(d_s, 2), 'delta_vs_m2_evidence': round(b1_s - sco['pt'], 2)},
              'replicate_gap_levels': lev['b1-r1'] - lev['b1-r2'],
              'per_game_sign_vs_controls': {'better': sum(1 for v in per_game.values() if v['delta'] > 0),
                                            'worse': sum(1 for v in per_game.values() if v['delta'] < 0),
                                            'tie': sum(1 for v in per_game.values() if v['delta'] == 0)},
              'pin_games': {k: sum(1 for x in v.values() if x.get('trims_with_pin', 0) > 0) for k, v in counters.items()},
              'transcripts': {k: {'exceptions': e, 'markers': m} for k, (e, m) in tx.items()},
              'per_game': per_game}
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    smoke(sys.argv[2]) if sys.argv[1] == 'smoke' else pair(sys.argv[2], sys.argv[3])
