"""How much of a hidden draw's score is run-level luck (shared by the 55 public and 55 private games)?
Model: s[g,r] = mu_g + u_r + e[g,r]. Fit the additive game x run model on full-length public-25 runs,
estimate sigma_u^2 (run) and sigma_e^2 (per game), then project to a 55-game half of the hidden set.
Read-only over benchmark.json files."""
import json, re, statistics as st, random
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
FULL = {  # full-length (~122 min) public-25 dfranzen-family runs
 'b1-r1': 'solvers/m2_b1/experiment/pair-1/b1-r1', 'b1-r2': 'solvers/m2_b1/experiment/pair-1/b1-r2',
 'ev-ctl': 'solvers/m2_evidence/experiment/pair-1/control', 'ev-trt': 'solvers/m2_evidence/experiment/pair-1/treatment',
 'e2-c14': 'analysis/model-serving-20261011/wave-1/e2-c14', 'b1-c14': 'analysis/model-serving-20261011/wave-2/b1-c14',
 'pr-ctl': 'analysis/pruning-transfer-20261011/control', 'pr-cand2': 'analysis/pruning-transfer-20261011/candidate2'}
SCREEN = {  # 60-min screens
 's1-b1ctl': 'solvers/m2_f1/experiment/screen-1/b1ctl', 's1-f1': 'solvers/m2_f1/experiment/screen-1/f1',
 's2-b1ctl': 'solvers/m2_f1/experiment/screen-2/b1ctl', 's2-f1x': 'solvers/m2_f1/experiment/screen-2/f1x',
 'lex-raw': 'solvers/m2_lexicon/experiment/screen-1/rawctl', 'lex': 'solvers/m2_lexicon/experiment/screen-1/lex'}
def load(p):
    runs = json.load(open(ROOT / p / 'output/benchmark.json'))['game_runs']
    out = {}
    for g in runs:
        m = re.search(r'tokens=(\d+)', g.get('solver_note') or '')
        out[g['game_id']] = (g['final_score'], g['levels_completed'], int(m.group(1)) if m else None)
    return out
def anova(runs, key=0):
    names = list(runs); games = sorted(set.intersection(*(set(runs[n]) for n in names)))
    X = [[runs[n][g][key] for n in names] for g in games]
    G, R = len(games), len(names)
    grand = st.mean(v for row in X for v in row)
    rm = [st.mean(X[i][j] for i in range(G)) for j in range(R)]
    gm = [st.mean(row) for row in X]
    ss_r = G * sum((m - grand) ** 2 for m in rm)
    ss_e = sum((X[i][j] - gm[i] - rm[j] + grand) ** 2 for i in range(G) for j in range(R))
    ms_r, ms_e = ss_r / (R - 1), ss_e / ((G - 1) * (R - 1))
    # E[ms_r] = sigma_e^2 + G sigma_u^2 ; E[ms_e] = sigma_e^2
    su2 = max(0.0, (ms_r - ms_e) / G)
    # permutation null: shuffle run labels within each game (exchangeable under no run effect)
    rng = random.Random(0); exceed = 0; N = 4000
    for _ in range(N):
        Y = [rng.sample(row, R) for row in X]
        rmp = [st.mean(Y[i][j] for i in range(G)) for j in range(R)]
        if G * sum((m - grand) ** 2 for m in rmp) >= ss_r: exceed += 1
    return dict(runs=names, G=G, run_means=dict(zip(names, [round(m, 2) for m in rm])), grand=round(grand, 2),
                ms_run=round(ms_r, 2), ms_resid=round(ms_e, 2), F=round(ms_r / ms_e, 2),
                perm_p=round((exceed + 1) / (N + 1), 4), sigma_e=round(ms_e ** .5, 2), sigma_u=round(su2 ** .5, 2),
                ratio_u2_over_e2=su2 / ms_e)
def project(a, n=55, obs_sd=None, obs_mean=None):
    r = a['ratio_u2_over_e2']; rho = r / (r + 1 / n)  # corr(public half, private half) of one run
    out = dict(n_games=n, corr_public_private=round(rho, 3))
    if obs_sd:  # alternative: take hidden sd as given, scale sigma_e from public by mean ratio
        s = obs_mean / a['grand']; e2n = (a['sigma_e'] * s) ** 2 / n
        u2 = max(0.0, obs_sd ** 2 - e2n)
        out.update(scaled_sigma_e=round(a['sigma_e'] * s, 2), predicted_sd_no_run_effect=round(e2n ** .5, 2),
                   implied_sigma_u=round(u2 ** .5, 2), corr_from_hidden_sd=round(u2 / obs_sd ** 2, 3))
    return out
if __name__ == '__main__':
    full = {k: load(v) for k, v in FULL.items()}; scr = {k: load(v) for k, v in SCREEN.items()}
    ctl = [23.87, 29.44, 24.97, 29.38, 31.31]
    res = {}
    for label, runs in [('full_all8', full), ('full_identical_b1', {k: full[k] for k in ['b1-r1', 'b1-r2']}),
                        ('screen_all6', scr), ('screen_identical_b1ctl', {k: scr[k] for k in ['s1-b1ctl', 's2-b1ctl']})]:
        a = anova(runs); a['projection'] = project(a, 55, st.stdev(ctl), st.mean(ctl)); res[label] = a
        # does run-level token throughput track run score?
        tok = {n: sum(v[2] or 0 for v in runs[n].values()) for n in runs}
        a['run_tokens_M'] = {n: round(t / 1e6, 2) for n, t in tok.items()}
        if len(runs) > 2:
            xs = [tok[n] for n in runs]; ys = [a['run_means'][n] for n in runs]
            a['corr_tokens_vs_score'] = round(st.correlation(xs, ys), 3)
    res['hidden_controls'] = dict(scores=ctl, mean=round(st.mean(ctl), 2), sd=round(st.stdev(ctl), 2))
    Path(__file__).with_name('RESULT.json').write_text(json.dumps(res, indent=1))
    for k, a in res.items():
        if k == 'hidden_controls': print(k, a); continue
        print(f"\n{k}: G={a['G']} F={a['F']} perm_p={a['perm_p']} sigma_e={a['sigma_e']} sigma_u={a['sigma_u']}")
        print('  run means', a['run_means']); print('  tokens M', a['run_tokens_M'], 'corr', a.get('corr_tokens_vs_score'))
        print('  projection', a['projection'])
