"""Simulate play.py's rolling fair-share allocator for a competition-sized run.

Pure re-implementation of admission_budget() from
deployment/kaggle/submission-20260910-v5/play.py:26-40 plus the per-game loop at
play.py:84-102. Read-only; no I/O beyond stdout.
"""
import json, math

def admission_budget(timeout_s, game_count, request_timeout_s):
    allowance = (timeout_s - 180) / game_count - 30
    budget = allowance - request_timeout_s - 15
    if budget < 0:
        raise ValueError('Insufficient runtime')
    return budget

def simulate(total=25140.0, games=110, request_timeout=90.0, overhead=10.0, cap=None):
    """overhead = observed (game wall - admission window) seconds per game."""
    started, walls, budgets = 0.0, [], []
    deadline = started + total
    now = started
    for i in range(games):
        remaining = deadline - now
        try:
            b = admission_budget(remaining, games - i, request_timeout)
        except ValueError:
            budgets.append(None); walls.append(0.0); continue
        if cap is not None:
            b = min(b, cap)
        budgets.append(b)
        wall = b + overhead
        walls.append(wall)
        now += wall
    return dict(budgets=budgets, walls=walls, used=now - started,
                unused=deadline - now)

out = {}
for oh in (3.54, 10.0, 17.63):
    s = simulate(overhead=oh)
    b = [x for x in s['budgets'] if x is not None]
    out[f'overhead_{oh}'] = {
        'admission_first': b[0], 'admission_10th': b[9], 'admission_50th': b[49],
        'admission_last': b[-1], 'mean_admission': sum(b)/len(b),
        'mean_game_wall': s['used']/110, 'total_used_s': s['used'],
        'unused_s': s['unused'], 'unused_fraction': s['unused']/25140,
    }
# observed public run for calibration
obs = dict(admission=75.0, walls=[92.62774050200005, 83.83993034100001, 78.54498956999987])
out['observed_overheads'] = [w - 75.0 for w in obs['walls']]
# action accounting: actions arise only from admitted model calls
out['observed'] = dict(
    actions=5, calls=34, gameplay_s=255.019576933,
    admission_s_total=225.0,
    actions_per_admission_second=5/225.0,
    actions_per_gameplay_second=5/255.019576933,
    calls_per_admission_second=34/225.0,
)
for oh in (3.54, 10.0, 17.63):
    k = f'overhead_{oh}'
    total_admission = out[k]['mean_admission']*110
    out[k]['total_admission_s'] = total_admission
    out[k]['actions_if_rate_per_admission_second'] = total_admission * (5/225.0)
    out[k]['actions_if_rate_per_gameplay_second_naive'] = 25140 * (5/255.019576933)
print(json.dumps(out, indent=2))
