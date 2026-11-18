"""Luby fixed-cutoff restart arithmetic on the empirical per-level action costs (levelcosts.json).
Cost per clear with cutoff tau = E[min(X,tau)] / P(X<=tau); a censored run counts as X>c.
Valid only if every censored c >= tau (else the level is skipped for that tau).
Treats a restart as a fresh independent draw of the whole level: an optimistic assumption."""
import json
d = json.load(open("levelcosts.json"))
TAUS = [15, 25, 40, 60]
print(f"{'level':9} n  clr  no-restart: P(clear) mean-cost-if-cleared | " + " | ".join(f"tau={t}: P  cost" for t in TAUS))
tot = {t: [0, 0] for t in TAUS}
for k, v in sorted(d.items()):
    if len(v) < 6: continue
    xs = [(a, ok) for a, ok, _ in v]; c = [a for a, ok in xs if ok]
    if not c: continue
    row = f"{k:9} {len(v)}  {len(c)}    {len(c)/len(v):.2f}  {sum(c)/len(c):6.0f}        | "
    cells = []
    for t in TAUS:
        if any((not ok) and a < t for a, ok in xs): cells.append("   n/a    "); continue
        p = sum(ok and a <= t for a, ok in xs) / len(xs)
        e = sum(min(a, t) for a, ok in xs) / len(xs)
        cells.append(f"{p:.2f} {e/p:5.0f}" if p else "0.00   inf")
    print(row + " | ".join(cells))
