"""Independent read-only recomputation of the coordinator's numeric claims.

Reads only raw/ artifacts under the run directory. Writes nothing.
Run:  python3 analysis/deployment-probes/20260910-budget-public-v1/claude-review/recheck.py
"""
import json, statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
Q = RUN / 'raw' / 'qualification'
rows = lambda p: [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
load = lambda p: json.loads(p.read_text())

out = {}
progress = rows(Q / 'game-progress.jsonl')
result = load(Q / 'result.json')
allocs = rows(Q / 'budget-allocations.jsonl')

# ---- claim 2: session times, actions, calls, levels -------------------------
out['games'] = [{k: r[k] for k in ('game_id', 'actions', 'solver_actions',
                                   'levels_completed', 'state', 'stop_reason',
                                   'game_elapsed_seconds', 'elapsed_seconds')}
                for r in progress]
out['sum_game_elapsed'] = sum(r['game_elapsed_seconds'] for r in progress)
out['launcher_elapsed_seconds'] = result['official_launcher']['elapsed_seconds']
out['last_game_cumulative_elapsed'] = progress[-1]['elapsed_seconds']

sessions = sorted(p for p in (Q / 'solver-workspaces').iterdir() if p.is_dir())
calls_by_session, allrows = {}, []
for s in sessions:
    cr = rows(s / 'ipc/model_calls.jsonl')
    calls_by_session[s.name] = cr
    for c in cr:
        allrows.append((s.name, c))
out['completed_calls_per_session'] = {k: len(v) for k, v in calls_by_session.items()}
out['total_calls'] = sum(len(v) for v in calls_by_session.values())

# ---- claim 3: model-call wall times ----------------------------------------
per_session_wall = {k: sum(c['wall_seconds'] for c in v) for k, v in calls_by_session.items()}
out['model_wall_per_session'] = per_session_wall
out['model_wall_total'] = sum(per_session_wall.values())
out['model_wall_fraction_of_gameplay'] = out['model_wall_total'] / out['launcher_elapsed_seconds']
w = sorted(c['wall_seconds'] for _, c in allrows)
out['wall_stats'] = {
    'n': len(w), 'mean': st.mean(w), 'median': st.median(w),
    'min': w[0], 'max': w[-1],
    'p90_linear_interp_numpy': None, 'p90_nearest_rank': None,
    'p90_exclusive': None, 'p90_statistics_quantiles_n10': None,
}
# nearest-rank: ceil(0.9*n)-th smallest
import math
out['wall_stats']['p90_nearest_rank'] = w[math.ceil(0.90 * len(w)) - 1]
# numpy / linear interpolation on (n-1) index
idx = 0.90 * (len(w) - 1)
lo, hi = math.floor(idx), math.ceil(idx)
out['wall_stats']['p90_linear_interp_numpy'] = w[lo] + (idx - lo) * (w[hi] - w[lo])
# exclusive (Excel PERCENTILE.EXC style): index = 0.9*(n+1)
idx2 = 0.90 * (len(w) + 1) - 1
lo2, hi2 = math.floor(idx2), math.ceil(idx2)
out['wall_stats']['p90_exclusive'] = w[lo2] + (idx2 - lo2) * (w[hi2] - w[lo2])
out['wall_stats']['p90_statistics_quantiles_n10'] = st.quantiles(w, n=10, method='inclusive')[-1]
out['wall_sorted_tail'] = w[-6:]

# ---- tokens ----------------------------------------------------------------
out['output_tokens_total'] = sum(c['completion_tokens'] for _, c in allrows)
out['reasoning_tokens_total'] = sum(c.get('reasoning_tokens') or 0 for _, c in allrows)
pt = [c['prompt_tokens'] for _, c in allrows]
out['prompt_tokens_range'] = [min(pt), max(pt)]
out['prompt_tokens_total'] = sum(pt)
out['summarizer_calls'] = sum(1 for _, c in allrows if c.get('summarizer'))
out['reasoning_present_calls'] = sum(1 for _, c in allrows if c.get('reasoning_present'))
out['purposes'] = sorted({c['request_purpose'] for _, c in allrows})
out['finish_reasons'] = {r: sum(1 for _, c in allrows if c['finish_reason'] == r)
                         for r in sorted({c['finish_reason'] for _, c in allrows})}

# ---- claim 4: notebook / outer / gameplay clocks ---------------------------
terminal = load(RUN / 'raw' / 'arc3-first-submission-20260910.log')
out['notebook_last_log_seconds'] = max(r['time'] for r in terminal)
out['notebook_first_log_seconds'] = min(r['time'] for r in terminal)
out['outer_elapsed_seconds'] = result['elapsed_s']
out['notebook_minus_gameplay'] = out['notebook_last_log_seconds'] - out['launcher_elapsed_seconds']
out['outer_minus_gameplay'] = out['outer_elapsed_seconds'] - out['launcher_elapsed_seconds']
out['notebook_minus_outer'] = out['notebook_last_log_seconds'] - out['outer_elapsed_seconds']

# ---- claim 6: arithmetic illustrations -------------------------------------
def illustration(total_gameplay_s, games=110):
    per = total_gameplay_s / games
    # observed short-run action throughput
    acts = sum(r['solver_actions'] for r in progress)
    gsec = sum(r['game_elapsed_seconds'] for r in progress)
    rate = acts / gsec
    return {'per_game_seconds': per, 'actions_per_second': rate,
            'actions_per_game': per * rate, 'total_actions': per * rate * games}
out['illustration_25140'] = illustration(25140)
out['illustration_32400_minus_overhead'] = illustration(32400 - out['notebook_minus_gameplay'])
out['observed_action_rate'] = (sum(r['solver_actions'] for r in progress)
                               / sum(r['game_elapsed_seconds'] for r in progress))

# ---- calls before first action ---------------------------------------------
pre = {}
for s in sessions:
    acts = rows(s / 'ipc/actions.jsonl') if (s / 'ipc/actions.jsonl').exists() else []
    cr = calls_by_session[s.name]
    # calls at turn 0/step 0 == before the first submitted batch executed
    pre[s.name] = {
        'calls_step0': sum(1 for c in cr if c['step'] == 0),
        'calls_by_step': {k: sum(1 for c in cr if c['step'] == k)
                          for k in sorted({c['step'] for c in cr})},
        'action_rows': len(acts),
    }
out['pre_first_action'] = pre
out['pre_first_action_total'] = sum(v['calls_step0'] for v in pre.values())

# ---- budget allocations ----------------------------------------------------
out['allocations'] = allocs
print(json.dumps(out, indent=2, default=str))
