"""Simulate the private board from the 10-18 public board under 'public best = max of N_eff draws'.
Each team: true mean m, per-draw sd = CV*m; N_eff = f*SubmissionCount (f<1 because early subs are
other code). Solve m from E[max of N_eff] = public best, then draw the private score of the two
picks (top-2 public), each m + corr*excess + sqrt(1-corr^2)*noise, take the better. Read-only."""
import csv, random, statistics as st, sys
from statistics import NormalDist
Z = NormalDist()
def emax(n):  # Blom approximation to E[max of n std normals]
    return 0.0 if n <= 1 else Z.inv_cdf((n - 0.375) / (n + 0.25))
rows = list(csv.DictReader(open('leaderboard-20261018.csv', encoding='utf-8-sig')))
teams = [(r['TeamName'], float(r['Score']), int(r['SubmissionCount'])) for r in rows if r['Score']]
US = 'Pranav Agrawal'; US_MEAN = 27.8
def run(f, cv, corr, seed=0, reps=200):
    rng = random.Random(seed); lines = {'gold': [], 'silver': [], 'bronze': []}; ranks = []
    n = len(teams); cut = {'gold': int(10 + .002 * n), 'silver': int(.05 * n), 'bronze': int(.1 * n)}
    for _ in range(reps):
        priv = []
        for name, best, k in teams:
            ne = max(1, round(f * k))
            m = US_MEAN if name == US else best / (1 + cv * emax(ne))
            s = cv * m
            pubs = sorted((rng.gauss(0, 1) for _ in range(ne)), reverse=True)[:2]
            p = max(m + s * (corr * z + (1 - corr ** 2) ** .5 * rng.gauss(0, 1)) for z in pubs)
            priv.append((p, name))
        priv.sort(reverse=True)
        for kk, c in cut.items(): lines[kk].append(priv[c - 1][0])
        ranks.append(next(i for i, (_, nm) in enumerate(priv, 1) if nm == US))
    return {k: round(st.median(v), 1) for k, v in lines.items()}, int(st.median(ranks))
print('f    cv    corr  private lines (median)                 our rank (mean 27.8)')
for f in [0.25, 0.5, 1.0]:
    for cv in [0.08, 0.115]:
        for corr in [0.15, 0.7]:
            L, r = run(f, cv, corr)
            print(f'{f:<4} {cv:<5} {corr:<5} {L}  {r}')
print('no inflation (f->0):', {k: v for k, v in zip(['gold', 'silver', 'bronze'], [sorted([b for _, b, _ in teams], reverse=True)[c - 1] for c in [18, 222, 445]])})
