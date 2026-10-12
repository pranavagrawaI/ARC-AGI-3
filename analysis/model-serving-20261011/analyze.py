"""Readout for wave-1 (EXPERIMENT-CARD.md). Mechanism first, then levels.

  python3 -B analyze.py <arm-output-dir> [label]     e.g. analyze.py wave-1/e1-swift/output E1

Controls: c1 = analysis/m2-winners-20261001/runs/out-control, pc = solvers/m2_evidence/experiment/pair-1/control/output.
Throughput is time-weighted: each SGLang "Decode batch" line weighs the seconds until the next one (capped at 10 s so
idle gaps don't count), over run-minutes 30-90 after the first decode line, the same estimator for every run.
"""
import datetime as dt, json, re, statistics as st, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
C1 = REPO / 'analysis/m2-winners-20261001/runs/out-control'
PC = REPO / 'solvers/m2_evidence/experiment/pair-1/control/output'
DECODE = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\] Decode batch, #running-req: (\d+), #full token: (\d+), "
                    r"full token usage: ([\d.]+).*?accept len: ([\d.]+).*?gen throughput \(token/s\): ([\d.]+), #queue-req: (\d+)")


def P(x): return json.loads(x) if isinstance(x, str) else x


def serve_stats(out, lo_min=30, hi_min=90):
    rows = []
    for line in (out / 'serve.log').read_text(errors='replace').splitlines():
        m = DECODE.match(line)
        if m:
            t = dt.datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S').timestamp()
            rows.append((t, int(m.group(2)), float(m.group(4)), float(m.group(5)), float(m.group(6)), int(m.group(7))))
    if not rows:
        return {'decode_lines': 0}
    t0 = rows[0][0]
    w_tok = w = 0.0
    acc, use, run, queue = [], [], [], []
    for (t, r, u, a, g, q), nxt in zip(rows, rows[1:]):
        if not (lo_min * 60 <= t - t0 < hi_min * 60):
            continue
        dt_s = min(10.0, max(0.0, nxt[0] - t))
        w_tok += g * dt_s; w += dt_s
        acc.append(a); use.append(u); run.append(r); queue.append(q)
    text = (out / 'serve.log').read_text(errors='replace')
    retract = sum(1 for l in text.splitlines() if 'retract' in l.lower() and 'server_args' not in l)
    s = lambda v: sorted(v)[int(0.9 * (len(v) - 1))] if v else None
    return {'decode_lines': len(rows), 'tw_gen_tok_s': round(w_tok / w, 1) if w else None, 'window_s': round(w),
            'accept_len_median': st.median(acc) if acc else None, 'running_median': st.median(run) if run else None,
            'kv_usage_median': st.median(use) if use else None, 'kv_usage_p90': s(use),
            'queue_median': st.median(queue) if queue else None, 'retract_lines': retract}


def bench(out):
    b = json.loads((out / 'benchmark.json').read_text())
    games = {}
    for g in b['game_runs']:
        h, apl = P(g['history']), P(g['actions_per_level'])
        done = int(g['levels_completed'] or 0)
        k, lv_tok = 0, {}
        for L in range(min(done, len(apl))):
            lv_tok[L] = sum(x.get('generated_tokens', 0) for x in h[k:k + apl[L]]); k += apl[L]
        games[g['game_id'][:4]] = {'levels': done, 'score': g.get('final_score') or 0.0, 'level_tokens': lv_tok,
                                   'tokens': sum(x.get('generated_tokens', 0) for x in h), 'state': g['state']}
    return games


def markers(out):
    logs = [p for p in out.glob('*.log') if p.name != 'serve.log']
    text = ''
    for p in logs:
        raw = p.read_text(errors='replace')
        try:
            text += ''.join(x.get('data', '') for x in json.loads(raw))
        except Exception:
            text += raw
    pats = [r'Swift view: \d+ files from \d+ parts', r'priority gate active: \d+ concurrent streams', r'READY after \d+s',
            r'DEADLINE at \d+s', r'M2 B1 installed', r'SGLang server exited during startup[^\n]*', r'Traceback \(most recent call last\)']
    return {p: sorted(set(re.findall(p, text)))[:3] for p in pats}


def main():
    out = Path(sys.argv[1]).resolve(); label = sys.argv[2] if len(sys.argv) > 2 else out.parent.name
    arm, c1, pc = bench(out), bench(C1), bench(PC)
    res = {'label': label, 'markers': markers(out), 'serve': {label: serve_stats(out), 'c1': serve_stats(C1), 'pc': serve_stats(PC)}}
    tot = lambda g, k: round(sum(v[k] for v in g.values()), 2)
    mean_score = lambda g: round(sum(v['score'] for v in g.values()) / len(g), 2)
    cm_lv = (tot(c1, 'levels') + tot(pc, 'levels')) / 2; cm_sc = (mean_score(c1) + mean_score(pc)) / 2
    res['levels'] = {label: tot(arm, 'levels'), 'c1': tot(c1, 'levels'), 'pc': tot(pc, 'levels'), 'control_mean': cm_lv,
                     'delta_vs_control_mean': tot(arm, 'levels') - cm_lv}
    res['mean_score'] = {label: mean_score(arm), 'c1': mean_score(c1), 'pc': mean_score(pc), 'control_mean': round(cm_sc, 2),
                         'delta_vs_control_mean': round(mean_score(arm) - cm_sc, 2)}
    res['n_games'] = len(arm)
    d = {g: arm[g]['levels'] - (c1[g]['levels'] + pc[g]['levels']) / 2 for g in arm if g in c1 and g in pc}
    res['per_game_vs_control_mean'] = {'+': sum(v > 0 for v in d.values()), '-': sum(v < 0 for v in d.values()),
                                       '0': sum(v == 0 for v in d.values()), 'by_game': d}
    ratios = [arm[g]['level_tokens'][L] / c1[g]['level_tokens'][L] for g in arm if g in c1
              for L in arm[g]['level_tokens'] if L in c1[g]['level_tokens'] and c1[g]['level_tokens'][L] > 0]
    res['tokens_per_cosolved_level_vs_c1'] = {'n': len(ratios), 'median_ratio': round(st.median(ratios), 3) if ratios else None}
    res['total_generated_tokens'] = {label: tot(arm, 'tokens'), 'c1': tot(c1, 'tokens'), 'pc': tot(pc, 'tokens')}
    print(json.dumps(res, indent=1))
    (out.parent / 'readout.json').write_text(json.dumps(res, indent=1) + '\n')


if __name__ == '__main__':
    main()
