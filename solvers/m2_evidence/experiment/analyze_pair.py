"""Apply pair-1's card to two output directories: invalidity first, then primary score, then levels, then
explanatory counters. python3 analyze_pair.py CONTROL_OUT TREATMENT_OUT [--skip-log-checks]"""
import glob, json, re, statistics, sys
from pathlib import Path

def log_text(out):
    logs = sorted(Path(out).glob('*.log'))
    return logs[0].read_text(errors='replace') if logs else ''

def arm(out):
    b = json.load(open(Path(out) / 'benchmark.json'))
    runs = {g['game_id'][:4]: g for g in b['game_runs']}
    return runs

def invalid(c_out, t_out, c, t, skip_logs):
    p = []
    for name, runs in (('control', c), ('treatment', t)):
        if len(runs) != 25:
            p.append(f'{name} has {len(runs)} games, not 25')
    if skip_logs:
        return p
    cl, tl = log_text(c_out), log_text(t_out)
    if 'M2 evidence installed' not in tl: p.append('treatment log lacks "M2 evidence installed"')
    if 'M2 evidence installed' in cl: p.append('control log contains the overlay')
    for name, text in (('control', cl), ('treatment', tl)):
        if 'max_runtime_s_per_game=7254' not in text: p.append(f'{name}: max_runtime_s_per_game != 7254')
        if 'concurrency=25' not in text: p.append(f'{name}: concurrency != 25')
        if 'priority gate active: 10 concurrent streams' not in text: p.append(f'{name}: priority gate != 10')
    return p

def evidence_counts(out):
    res = {}
    for f in glob.glob(str(Path(out) / 'artifacts/*_evidence.json')):
        d = json.load(open(f)); e = d['entries']
        res[Path(f).name[:4]] = {'entries': len(e), 'revisions': sum(1 for x in e if x.get('revises')), 'reshows': d.get('reshows', 0)}
    return res

def main():
    c_out, t_out = sys.argv[1], sys.argv[2]
    c, t = arm(c_out), arm(t_out)
    problems = invalid(c_out, t_out, c, t, '--skip-log-checks' in sys.argv)
    games = sorted(set(c) & set(t))
    sc = {g: (c[g]['final_score'], t[g]['final_score']) for g in games}
    lv = {g: (c[g]['levels_completed'], t[g]['levels_completed']) for g in games}
    d_score = statistics.mean(x[1] for x in sc.values()) - statistics.mean(x[0] for x in sc.values())
    d_lev = sum(x[1] for x in lv.values()) - sum(x[0] for x in lv.values())
    ev = evidence_counts(t_out)
    used = sum(1 for v in ev.values() if v['entries'] > 0)
    if problems: verdict = 'INVALID'
    elif d_score >= 10 and d_lev >= 10: verdict = 'POSITIVE'
    elif d_score <= -10 and d_lev <= -10: verdict = 'NEGATIVE'
    else: verdict = 'INCONCLUSIVE'
    report = {
        'verdict': verdict, 'invalid_reasons': problems,
        'treatment_delivered': used >= 5, 'games_with_evidence': used,
        'mean_score': {'control': round(statistics.mean(x[0] for x in sc.values()), 2), 'treatment': round(statistics.mean(x[1] for x in sc.values()), 2), 'delta': round(d_score, 2)},
        'levels': {'control': sum(x[0] for x in lv.values()), 'treatment': sum(x[1] for x in lv.values()), 'delta': d_lev},
        'score_sign': {'better': sum(1 for a, b in sc.values() if b > a), 'worse': sum(1 for a, b in sc.values() if b < a), 'tie': sum(1 for a, b in sc.values() if a == b)},
        'per_game': {g: {'score': [round(sc[g][0], 1), round(sc[g][1], 1)], 'levels': list(lv[g]), 'evidence': ev.get(g)} for g in games},
    }
    print(json.dumps(report, indent=1))

if __name__ == '__main__':
    main()
