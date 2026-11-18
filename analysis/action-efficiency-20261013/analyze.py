"""Offline action accounting. Writes only beside this script. Run with python -B.

No model calls, training, launches, or submission. All level indices in artifacts are zero-based.
"""
import ast
import collections
import hashlib
import json
import math
import struct
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
ENV = Path('/home/pranav/projects/arc-agi-3-reference/competition-env/environment_files')
RUNS = {
    'control': ('analysis/m2-winners-20261001/runs/out-control', 'original incumbent control'),
    'control-r2': ('analysis/m2-winners-20261001/runs/out-control-r2', 'incumbent repeat; shorter horizon'),
    'pruning-control': ('analysis/pruning-transfer-20261011/control/output', 'later c14 reference control'),
    'c14': ('analysis/model-serving-20261011/wave-1/e2-c14/output', '14-stream serving variant'),
    'b1-c14': ('analysis/model-serving-20261011/wave-2/b1-c14/output', 'B1 prompt/history plus 14-stream variant'),
}


def parse(v):
    return ast.literal_eval(v) if isinstance(v, str) else v


def level_score(b, a):
    return min(115.0, 100 * (b / a) ** 2) if a > 0 else 0.0


def game_score(bases, actions, completed):
    weights = len(bases) * (len(bases) + 1) / 2
    raw = sum((i + 1) * level_score(bases[i], actions[i]) for i in range(completed)) / weights
    return min(raw, 100 * completed * (completed + 1) / 2 / weights)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def bh(board, interior=False):
    # Matches the shipped harness gameplay_changed mask: strip all four boundary rows/columns.
    if interior:
        board = [r[1:-1] for r in board[1:-1]]
    return hashlib.sha256(bytes(v for row in board for v in row)).hexdigest()


def main():
    graphs = {}
    manifest = []
    for p in sorted((ROOT / 'analysis/oracle-progress-20261009/graphs').glob('*.json')):
        g = json.loads(p.read_text())
        graphs[g['summary']['game'], g['summary']['level']] = (g, p)
    all_rows, game_rows, action_rows, candidates, summaries = [], [], [], [], []
    for name, (rel, role) in RUNS.items():
        out = ROOT / rel
        bp = out / 'benchmark.json'
        benchmark = json.loads(bp.read_text())
        manifest.append(dict(path=str(bp.relative_to(ROOT)), sha256=sha(bp)))
        rows, games, audit = [], [], collections.Counter()
        for r in benchmark['game_runs']:
            gid = r['game_id']
            bases, apl, hist = map(parse, (r['base_actions_per_level'], r['actions_per_level'], r['history']))
            done = int(r['levels_completed'] or 0)
            metadata = ENV / gid[:4] / gid[5:] / 'metadata.json'
            assert bases == json.loads(metadata.read_text())['baseline_actions'], metadata
            ep = out / 'artifacts' / f'{gid}_p0_events.jsonl'
            manifest.append(dict(path=str(ep.relative_to(ROOT)), sha256=sha(ep)))
            events = [json.loads(l) for l in ep.open()]
            actions = [e for e in events if e['type'] == 'action']
            assert len(actions) == len(hist) == sum(apl), (name, gid, len(actions), len(hist), sum(apl))
            assert [e['action_num'] for e in actions] == list(range(1, len(actions) + 1))
            assert all(e['action_name'] == h['action']['id'] for e, h in zip(actions, hist))
            assert sum(bool(e.get('level_completed')) for e in actions) == done - (r['state'] == 'won')
            assert max(e['score'] for e in events) == done  # event score is LEVEL COUNT, not efficiency score!
            previous = next(e for e in events if e['type'] == 'initial')['board']
            before, level = [], 0
            seen_noops, seen_transitions = set(), set()
            for e, h in zip(actions, hist):
                before.append(previous)
                a = h['action']
                pfull, pinner = bh(previous), bh(previous, True)
                full, inner = bh(e['board']), bh(e['board'], True)
                key = (level, pinner, json.dumps(a, sort_keys=True))
                noop = pinner == inner and not e.get('level_completed') and not e.get('run_complete')
                known_noop = noop and key in seen_noops
                repeated = key in seen_transitions
                record = dict(run=name, game_id=gid, level=level, action_num=e['action_num'],
                              analysis_step=e.get('analysis_step'), action=a, interior_unchanged=noop,
                              full_frame_unchanged=pfull == full, repeated_observed_noop=known_noop,
                              repeated_visual_transition=repeated, reset=a['id'] == 'RESET',
                              fatal=bool(e.get('game_over')), level_completed=bool(e.get('level_completed')),
                              run_complete=bool(e.get('run_complete')), solved_level=level < done,
                              source=str(ep.relative_to(ROOT)))
                action_rows.append(record)
                if level < done:
                    for field in ('interior_unchanged', 'full_frame_unchanged', 'repeated_observed_noop',
                                  'repeated_visual_transition', 'reset', 'fatal'):
                        audit[field] += record[field]
                if noop:
                    seen_noops.add(key)
                seen_transitions.add(key)
                previous = e['board']
                if e.get('level_completed') or e.get('run_complete'):
                    level += 1
            assert level == done, (name, gid, level, done)
            k = 0
            grows = []
            for lv in range(done):
                n, base = apl[lv], bases[lv]
                seg = action_rows[-len(actions) + k:-len(actions) + k + n] if k+n < len(actions) else action_rows[-len(actions)+k:]
                assert len(seg) == n and all(x['level'] == lv for x in seg)
                resets = [i for i, x in enumerate(seg) if x['reset']]
                last_reset = resets[-1] + 1 if resets else 0
                graph, gp = graphs.get((gid, lv), (None, None))
                pixels = [v for row in before[k] for v in row]
                root_match = graph is not None and hashlib.md5(struct.pack('<'+'q'*len(pixels), *pixels)).hexdigest() == graph['root']
                droot = graph['summary']['d_root'] if graph and root_match else None
                exact = bool(graph and root_match and graph['summary']['exhausted'] and droot is not None)
                row = dict(run=name, game_id=gid, level=lv, display_level=lv+1, actions=n, baseline_actions=base,
                           level_score=level_score(base, n), baseline_efficiency_level_score=100.0,
                           optimistic_capped_level_score=115.0, weight=lv+1,
                           start_action_num=k+1, end_action_num=k+n, resets=len(resets),
                           deaths=sum(x['fatal'] for x in seg), reset_episode_prefix_actions=last_reset,
                           interior_unchanged=sum(x['interior_unchanged'] for x in seg),
                           full_frame_unchanged=sum(x['full_frame_unchanged'] for x in seg),
                           repeated_observed_noop=sum(x['repeated_observed_noop'] for x in seg),
                           repeated_visual_transition=sum(x['repeated_visual_transition'] for x in seg),
                           graph_exhausted_with_matching_root=exact, oracle_actions=droot if exact else None,
                           oracle_level_score=level_score(base, droot) if exact else None,
                           graph_shortest_found_actions=droot,
                           oracle_scope='exact only under prior frame-key/click-candidate assumptions' if exact else 'unknown',
                           graph_path=str(gp.relative_to(ROOT)) if gp else None,
                           evidence=dict(benchmark=str(bp.relative_to(ROOT)), events=str(ep.relative_to(ROOT)),
                                         metadata=str(metadata), history_slice=[k, k+n]))
                rows.append(row); grows.append(row); k += n
            measured = game_score(bases, apl, done)
            assert abs(measured - r['final_score']) < 1e-9, (name, gid, measured, r['final_score'])
            ceiling = game_score(bases, [1]*len(bases), done)
            half_all = game_score(bases, [max(1, math.ceil(a/2)) for a in apl], done)
            exact_actions = list(apl)
            for row in grows:
                if row['oracle_actions'] is not None:
                    exact_actions[row['level']] = row['oracle_actions']
            # E <= A. Halving E can remove at most floor(A/2) integer actions.
            half_exploration_upper = half_all
            gr = dict(run=name, game_id=gid, levels_completed=done, total_levels=len(bases), actual_score=measured,
                      fixed_completions_ceiling=ceiling, perfect_efficiency_gain=ceiling-measured,
                      baseline_efficiency_score=game_score(bases, bases, done),
                      all_actions_halved_score=half_all,
                      exploration_halved_most_generous_score_upper=half_exploration_upper,
                      known_plan_at_entry_optimal_execution_score_upper=ceiling,
                      prior_exact_graphs_only_score=game_score(bases, exact_actions, done),
                      benchmark=str(bp.relative_to(ROOT)))
            games.append(gr)
            # Save model-origin plan/exploration snippets on SOLVED levels; no system/user boilerplate.
            numlevel = {}
            for row in grows:
                for num in range(row['start_action_num'], row['end_action_num']+1):
                    numlevel[num] = row['level']
            for e in events:
                if e['type'] != 'analysis':
                    continue
                text = e.get('transcript', '')
                blocks = list(__import__('re').finditer(r'\[(THINKING|TOOL CALL: python|TOOL RESULT: python)\]\n(.*?)(?=\n\[[A-Z][A-Z _:]*(?:python)?\]|\Z)', text, __import__('re').S))
                body = '\n'.join(m.group(0) for m in blocks)
                lv = numlevel.get(e['action_num'])
                if lv is not None and __import__('re').search(r'\b(BFS|shortest|solution|path|plan|reset|no.op)\b', body, __import__('re').I):
                    candidates.append(dict(run=name, game_id=gid, level=lv, analysis_step=e['analysis_step'],
                                           action_num_at_turn_end=e['action_num'], body=body,
                                           source=str(ep.relative_to(ROOT))))
        all_rows.extend(rows); game_rows.extend(games)
        count = len(games)
        summary = dict(run=name, role=role, directory=rel, games=count, solved_levels=len(rows),
                       solved_actions=sum(x['actions'] for x in rows),
                       levels_capped_at_115=sum(x['level_score'] == 115 for x in rows),
                       games_at_completion_ceiling=sum(x['perfect_efficiency_gain'] < 1e-9 for x in games),
                       waste_observations=dict(audit), reset_episode_prefix_actions=sum(x['reset_episode_prefix_actions'] for x in rows),
                       exact_graph_level_observations=sum(x['oracle_actions'] is not None for x in rows),
                       means={field:sum(g[field] for g in games)/count for field in
                              ('actual_score','fixed_completions_ceiling','perfect_efficiency_gain',
                               'baseline_efficiency_score','all_actions_halved_score',
                               'exploration_halved_most_generous_score_upper',
                               'known_plan_at_entry_optimal_execution_score_upper','prior_exact_graphs_only_score')})
        summary['max_direct_relative_gain_percent'] = 100*summary['means']['perfect_efficiency_gain']/summary['means']['actual_score']
        summaries.append(summary)
    for filename, records in [('levels.jsonl', all_rows), ('games.jsonl', game_rows),
                              ('actions.jsonl', action_rows), ('plan-candidates.jsonl', candidates)]:
        (HERE / filename).write_text(''.join(json.dumps(x) + '\n' for x in records))
    (HERE / 'input-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (HERE / 'accounting.json').write_text(json.dumps(dict(runs=summaries, validations=dict(
        benchmark_score_reconciled=True, event_actions_reconciled=True,
        won_final_level_reconciled=True, baselines_equal_game_metadata=True)), indent=2)+'\n')
    for s in summaries:
        print(json.dumps(s))


if __name__ == '__main__':
    main()
