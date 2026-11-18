"""Offline Arm P readout. Never launches, changes, or submits a Kaggle run.

python3 analysis/arm-p-full-20260930/assess.py
Missing outputs leave the decision pending; invalid outputs cannot promote P.
"""
import argparse
import ast
import bisect
import datetime as dt
import hashlib
import json
import re
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = Path('/home/pranav/projects/arc-agi-3-reference/duck-qwen38/runs/flash-next')
ARMS = {'KF': 'out-kf2', 'C': 'out-c1', 'P-probe': 'out-p-probe', 'P1': 'out-p1', 'P2': 'out-p2'}
WINDOWS = [(0, 1200), (1200, 3960), (3960, 7920)]


def med(xs):
    return st.median(xs) if xs else None


def progression(run, events):
    """Join post-action scores to recorded cumulative game time, checking every action."""
    actions = [e for e in events if e['type'] == 'action']
    hist = run['history']
    if [e['action_num'] for e in actions] != list(range(1, len(hist) + 1)):
        raise ValueError('event/history action counts or numbering disagree')
    times = [h['wallclock_seconds'] for h in hist]
    if times != sorted(times):
        raise ValueError('history time went backwards')
    scores = [e['score'] for e in actions]
    if scores != sorted(scores) or (scores[-1] if scores else 0) != run['levels_completed']:
        raise ValueError('event progression disagrees with terminal benchmark')
    points = [(0.0, 0, 0)] + [(t, i, s) for i, (t, s) in enumerate(zip(times, scores), 1)]
    clears = [(t, i, s) for (t, i, s), prev in zip(points[1:], points) if s > prev[2]]
    return points, clears


def at_time(points, seconds):
    return points[bisect.bisect_right([p[0] for p in points], seconds) - 1]


def serving(folder, start):
    text = (folder / 'vllm-openai-server.log').read_text(errors='replace')
    rows, accepts = [], []
    for line in text.splitlines():
        stamp = re.search(r'(\d\d-\d\d \d\d:\d\d:\d\d)', line)
        if not stamp:
            continue
        t = (dt.datetime.strptime(f'{start.year}-{stamp[1]}', '%Y-%m-%d %H:%M:%S') - start).total_seconds()
        m = re.search(r'Avg generation throughput: ([\d.]+) tokens/s, Running: (\d+) reqs, Waiting: (\d+) reqs, GPU KV cache usage: ([\d.]+)%', line)
        if m:
            gen, running, waiting, kv = map(float, m.groups())
            rows.append({'seconds': t, 'gen_tok_s': gen, 'running': running, 'waiting': waiting, 'kv_percent': kv})
        m = re.search(r'Accepted: (\d+) tokens, Drafted: (\d+) tokens', line)
        if m:
            accepts.append((t, int(m[1]), int(m[2])))
    def summary(samples):
        return {'samples': len(samples), **{k: med([s[k] for s in samples]) for k in ['running', 'waiting', 'gen_tok_s', 'kv_percent']}}
    windows = []
    for lo, hi in WINDOWS:
        samples = [s for s in rows if lo <= s['seconds'] < hi]
        a = [(n, d) for t, n, d in accepts if lo <= t < hi]
        windows.append({'start_s': lo, 'end_s': hi, 'all': summary(samples),
                        'active_ge20': summary([s for s in samples if s['running'] + s['waiting'] >= 20]),
                        'waiting_positive': summary([s for s in samples if s['waiting'] > 0]),
                        'mtp_acceptance': sum(n for n, d in a) / sum(d for n, d in a) if sum(d for n, d in a) else None})
    prom = (folder / 'vllm-metrics-final.prom').read_text()
    latency = {}
    for name in ['e2e_request_latency_seconds', 'request_queue_time_seconds', 'request_prefill_time_seconds', 'request_decode_time_seconds']:
        vals = {suffix: sum(map(float, re.findall(r'^vllm:' + name + '_' + suffix + r'\{[^\n]*\} ([\d.e+\-]+)', prom, re.M))) for suffix in ['sum', 'count']}
        latency[name] = {**vals, 'mean': vals['sum'] / vals['count'] if vals['count'] else None}
    kv = re.search(r'GPU KV cache size: ([\d,]+) tokens', text)
    memory = re.findall(r'Model loading took ([\d.]+) GiB', text)
    plugin = re.findall(r'FLASHNEXT_PRUNE pid=\d+ load_weights totals (\{[^}]*\}) blocks=(\d+) keep=(\d+)', text)
    busy = [s for s in rows if 0 <= s['seconds'] < 7920 and s['running'] + s['waiting'] >= 20]
    return {'kv_tokens': int(kv[1].replace(',', '')) if kv else None,
            'model_gib': list(map(float, memory)), 'plugin': plugin,
            'busy': summary(busy), 'windows': windows, 'request_latency_whole_run': latency,
            'late_request_latency': None,
            'late_request_latency_limitation': 'Only final cumulative Prometheus histograms retained; per-request logging is disabled in frozen notebooks. Inter-action gaps are a separate measure.',
            'samples': rows}


def summarize(folder, name):
    bench = json.loads((folder / 'benchmark.json').read_text())
    runs = bench['game_runs']
    issues = []
    if len(runs) != 25 or len({r['game_id'] for r in runs}) != 25 or bench['n_passes'] != 1:
        issues.append('Expected exactly 25 distinct games and one pass')
    start = dt.datetime.fromisoformat(bench['start_time'])
    per_game = {}
    end = 1200 if name == 'P-probe' else 7920
    ticks = list(range(0, end, 300)) + [end]
    curve = {t: {'seconds': t, 'levels': 0, 'actions': 0} for t in ticks}
    sources = [folder / x for x in ['benchmark.json', 'score.json', 'vllm-openai-server.log', 'vllm-metrics-final.prom']]
    for r in runs:
        path = folder / 'artifacts' / (r['game_id'] + '_p0_events.jsonl')
        sources.append(path)
        events = [json.loads(line) for line in path.open()]
        points, clears = progression(r, events)
        for t in ticks:
            _, actions, levels = at_time(points, t)
            curve[t]['actions'] += actions
            curve[t]['levels'] += levels
        gaps = [(cur[0], cur[0] - prev[0]) for prev, cur in zip(points, points[1:])]
        act = [e for e in events if e['type'] == 'action']
        if r['state'] not in ['gave_up', 'won', 'completed', 'lost']:
            issues.append(f"{r['game_id']}: unexpected terminal state {r['state']}")
        per_game[r['game_id']] = {'levels': r['levels_completed'], 'actions': len(r['history']),
            'state': r['state'], 'wallclock_s': r['final_wallclock_seconds'],
            'first_action_s': points[1][0] if len(points) > 1 else None,
            'clears': [{'seconds': t, 'action': i, 'levels': s} for t, i, s in clears],
            'unchanged_board_actions': sum(not e.get('board_changed', True) for e in act),
            'inter_action_gap_medians': [{'start_s': lo, 'end_s': hi, 'seconds': med([gap for t, gap in gaps if lo <= t < hi])} for lo, hi in WINDOWS]}
    logs = list(folder.glob('duck-flash-next-*.log'))
    notebook_log = '\n'.join(f.read_text(errors='replace') for f in logs)
    sources.extend(logs)
    if not re.search(r'PUBLIC25_AUDIT runs=25 actions=\d+', notebook_log):
        issues.append('Missing final PUBLIC25_AUDIT')
    if f'PUBLIC25_SETTINGS budget_s={float(end)}' not in notebook_log:
        issues.append('Missing or different configured game horizon')
    measured = serving(folder, start)
    if name.startswith('P'):
        if 'ARM P LIVE' not in notebook_log or not measured['plugin']:
            issues.append('Pruning guard not confirmed')
        if not measured['model_gib'] or max(measured['model_gib']) >= 76:
            issues.append('Expected pruned model below 76 GiB')
        if measured['kv_tokens'] != 252658:
            issues.append('Unexpected KV pool')
        for totals, blocks, keep in measured['plugin']:
            counts = ast.literal_eval(totals)
            if blocks != '48' or keep != '448' or any(counts.get(k) != v for k, v in {'dropped': 36864, 'renumbered': 258048, 'routers': 48}.items()):
                issues.append('Pruning tensor counts disagree with qualified probe')
    teardown = json.loads((folder / 'vllm-server-teardown.json').read_text())
    sources.append(folder / 'vllm-server-teardown.json')
    score = json.loads((folder / 'score.json').read_text())
    return {'folder': str(folder), 'issues': issues, 'games': len(runs),
            'levels': sum(r['levels_completed'] for r in runs),
            'actions': sum(len(r['history']) for r in runs),
            'mean_score': st.mean(v['score'] for v in score['games'].values()),
            'curve': list(curve.values()), 'per_game': per_game, 'serving': measured,
            'shutdown_ok': teardown.get('shutdown_ok'),
            'sources': [{'path': str(f), 'sha256': hashlib.sha256(f.read_bytes()).hexdigest()} for f in sources]}


def decision(arms):
    if not all(k in arms for k in ['KF', 'C', 'P1', 'P2']):
        return {'status': 'PENDING', 'candidate': 'KF', 'reason': 'Both complete P passes are required; no missing pass is scored as zero.'}
    issues = [f'{k}: {issue}' for k in ['KF', 'C', 'P1', 'P2'] for issue in arms[k]['issues']]
    game_sets = [set(arms[k]['per_game']) for k in ['KF', 'C', 'P1', 'P2']]
    if any(s != game_sets[0] for s in game_sets):
        issues.append('Game sets differ')
    if arms['KF']['levels'] != 44 or arms['C']['levels'] != 42:
        issues.append('Control artifacts disagree with registered baseline')
    if issues:
        return {'status': 'INVALID', 'candidate': 'KF', 'issues': issues}
    mean = st.mean(arms[k]['levels'] for k in ['P1', 'P2'])
    mechanism = all((arms[k]['serving']['busy']['running'] or 0) >= 7 for k in ['P1', 'P2'])
    passes = mean >= 51.5 and mechanism
    return {'status': 'PASS' if passes else 'FAIL', 'candidate': 'P' if passes else 'KF',
            'mean_p': mean, 'control_mean': 43, 'delta': mean - 43,
            'threshold': 51.5, 'mechanism_pass': mechanism,
            'interpretation': 'Bundled pruning plus serving comparison; not an isolated causal estimate of speed.'}


def comparison(arms):
    if not all(k in arms for k in ['KF', 'C', 'P1', 'P2']):
        return None
    rows = []
    for g in sorted(arms['KF']['per_game']):
        row = {'game': g}
        if any(g not in arms[k]['per_game'] for k in ['KF', 'C', 'P1', 'P2']):
            continue
        for metric in ['levels', 'actions']:
            for k in ['KF', 'C', 'P1', 'P2']:
                row[f'{k}_{metric}'] = arms[k]['per_game'][g][metric]
            row[f'delta_{metric}'] = (row[f'P1_{metric}'] + row[f'P2_{metric}'] - row[f'KF_{metric}'] - row[f'C_{metric}']) / 2
        rows.append(row)
    return {'per_game': rows, 'games_gained': sum(r['delta_levels'] > 0 for r in rows),
            'games_lost': sum(r['delta_levels'] < 0 for r in rows),
            'net_gain_without_best_game': sum(r['delta_levels'] for r in rows) - max(r['delta_levels'] for r in rows),
            'note': 'Descriptive robustness checks, not additional promotion gates.'}


def render(result, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    colors = {'KF': '#697386', 'C': '#9aa5b5', 'P-probe': '#9c6bc0', 'P1': '#007c91', 'P2': '#d66c20'}
    for name, arm in result['arms'].items():
        c = arm['curve']
        for ax, metric in zip(axes[:2], ['levels', 'actions']):
            ax.step([x['seconds'] / 60 for x in c], [x[metric] for x in c], where='post', label=name, color=colors[name])
        rows = arm['serving']['samples']
        buckets = {}
        for s in rows:
            if s['seconds'] >= 0 and s['waiting'] > 0:
                buckets.setdefault(int(s['seconds'] // 300), []).append(s['running'])
        axes[2].plot([(i + .5) * 5 for i in sorted(buckets)], [med(buckets[i]) for i in sorted(buckets)], label=name, color=colors[name])
    for ax, title, ylabel in zip(axes, ['Solved levels', 'Executed actions', 'Concurrency while requests wait'], ['Levels', 'Actions', 'Median running requests / 5 min']):
        ax.set(title=title, xlabel='Minutes of play', ylabel=ylabel)
        ax.grid(alpha=.2)
    axes[0].legend()
    fig.suptitle('Arm P — ' + result['decision']['status'] + ' · totals use complete runs; curves use elapsed game time')
    fig.tight_layout()
    fig.savefig(output / 'progress.png', dpi=150)
    plt.close(fig)


def write_report(result, output):
    d = result['decision']
    lines = [f"# Arm P full-pass readout: {d['status']}", '',
             f"Local candidate: **{d['candidate']}**. No competition submission is performed.", '',
             '| Arm | Levels | Actions | Whole-run mean request latency (s) | Busy median running |',
             '|---|---:|---:|---:|---:|']
    for name, a in result['arms'].items():
        latency = a['serving']['request_latency_whole_run']['e2e_request_latency_seconds']['mean']
        lines.append(f"| {name} | {a['levels']} | {a['actions']} | {latency:.1f} | {a['serving']['busy']['running']} |")
    lines += ['', 'Rule: both full P passes valid; mean(P1,P2) >=51.5 levels against KF/C mean 43; pruning live and busy median running >=7 in each P pass.',
              '', '![Progress](progress.png)', '',
              'Curves join post-action event scores to cumulative benchmark history times and sample at five-minute intervals plus the horizon. Final totals include every recorded action; a final action may finish just beyond the nominal horizon. Concurrency uses log time relative to benchmark start. Busy means running plus waiting >=20. The graph conditions on waiting >0.', '',
              'Late individual request latency is unavailable: frozen notebooks retain only final cumulative histograms. Per-game inter-action gaps in assessment.json are a different measure.', '',
              'The comparison changes pruning and serving together. A null result does not isolate whether pruning lost capability or extra actions failed to help. Historical controls are not simultaneous randomized controls. The registered threshold is a decision rule, not a precisely calibrated significance test.', '']
    c = result.get('comparison')
    if c:
        lines += [f"Games gaining levels: {c['games_gained']}; losing: {c['games_lost']}. Net mean level gain after removing the best game: {c['net_gain_without_best_game']:+g}. These are diagnostics, not new gates.", '',
                  '| Game | KF | C | P1 | P2 | Mean level gain | Mean action gain |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for r in c['per_game']:
            lines.append(f"| {r['game']} | {r['KF_levels']} | {r['C_levels']} | {r['P1_levels']} | {r['P2_levels']} | {r['delta_levels']:+g} | {r['delta_actions']:+g} |")
    lines += ['', 'Operational checks:', '']
    for name, a in result['arms'].items():
        lines.append(f"- {name}: interpretation issues={a['issues']}; teardown shutdown_ok={a['shutdown_ok']}. Teardown is reported separately from completed gameplay.")
    for name, error in result['unavailable'].items():
        lines.append(f'- {name}: unavailable: {error}')
    lines += ['', 'Reproduce: `python3 analysis/arm-p-full-20260930/assess.py`. Raw source hashes, measurements, and all per-game clear times are in [assessment.json](assessment.json).', '']
    (output / 'REPORT.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--output', type=Path, default=HERE)
    parser.add_argument('--terminal', action='store_true', help='Both remote runs are terminal; missing or malformed artifacts are invalid, not pending.')
    args = parser.parse_args()
    arms, unavailable = {}, {}
    for name, folder in ARMS.items():
        try:
            arms[name] = summarize(args.root / folder, name)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            unavailable[name] = str(exc)
    result = {'generated_at': dt.datetime.now(dt.timezone.utc).isoformat(), 'arms': arms,
              'unavailable': unavailable, 'decision': decision(arms), 'comparison': comparison(arms)}
    required_missing = {k: v for k, v in unavailable.items() if k in ['KF', 'C', 'P1', 'P2']}
    if args.terminal and required_missing:
        result['decision'] = {'status': 'INVALID', 'candidate': 'KF', 'reason': 'Terminal run artifacts incomplete or inconsistent.', 'issues': required_missing}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'assessment.json').write_text(json.dumps(result, indent=2) + '\n')
    render(result, args.output)
    write_report(result, args.output)
    print(json.dumps({'decision': result['decision'], 'totals': {k: {x: a[x] for x in ['levels', 'actions', 'issues']} for k, a in arms.items()}, 'unavailable': unavailable}, indent=2))


if __name__ == '__main__':
    main()
