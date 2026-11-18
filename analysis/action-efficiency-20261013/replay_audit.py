"""Replay receipts and verify hindsight deletions. Run using arcengine venv Python -B.

Visual no-change is a diagnostic candidate, not permission to suppress a live action.
"""
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from arcengine import ARCBaseGame, ActionInput, GameAction, GameState
from analyze import HERE, ROOT, ENV, RUNS, parse, game_score


def engine(gid):
    p = ENV / gid[:4] / gid[5:] / (gid[:4]+'.py')
    spec = importlib.util.spec_from_file_location(gid[:4], p)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cls = next(c for c in vars(module).values() if isinstance(c, type) and issubclass(c, ARCBaseGame) and c is not ARCBaseGame)
    return cls()


def act(g, a):
    return g.perform_action(ActionInput(id=GameAction[a['id']], data=a.get('data') or {}))


def trial(g0, seq, lv):
    g = copy.deepcopy(g0)
    for i, a in enumerate(seq):
        fr = act(g, a)
        if fr.levels_completed > lv:
            return dict(solves=True, completion_action=i+1, exact_end=i+1 == len(seq))
        if fr.state == GameState.GAME_OVER:
            # Logged successful segments can intentionally include death followed by automatic RESET.
            # A deletion fails only if the remaining recorded sequence cannot reset and continue.
            if i+1 < len(seq) and seq[i+1]['id'] == 'RESET':
                continue
            return dict(solves=False, death_action=i+1)
    return dict(solves=False, exhausted_actions=len(seq))


def main():
    level_rows = {(r['run'], r['game_id'], r['level']):r for r in map(json.loads, (HERE/'levels.jsonl').open())}
    observations = {(r['run'], r['game_id'], r['action_num']):r for r in map(json.loads, (HERE/'actions.jsonl').open())}
    replay_checks, deletions, oracle_traces, run_summaries = [], [], [], []
    for name, (rel, role) in RUNS.items():
        records = json.loads((ROOT/rel/'benchmark.json').read_text())['game_runs']
        scores = {key:[] for key in ('remove_reset_prefix','remove_interior_unchanged', 'remove_repeated_noops')}
        totals = {key:0 for key in scores}
        for r in records:
            gid = r['game_id']; done = int(r['levels_completed'] or 0)
            hist, apl, bases = map(parse, (r['history'],r['actions_per_level'],r['base_actions_per_level']))
            g = engine(gid); fr = act(g, {'id':'RESET'})
            k = 0; counterfactual = {key:list(apl) for key in scores}
            for lv in range(done):
                row = level_rows[name,gid,lv]
                start = copy.deepcopy(g)
                seq = [h['action'] for h in hist[k:k+apl[lv]]]
                obs = [observations[name,gid,i] for i in range(k+1,k+1+apl[lv])]
                cases = {
                    'remove_reset_prefix':seq[row['reset_episode_prefix_actions']:],
                    'remove_interior_unchanged':[a for a,o in zip(seq,obs) if not o['interior_unchanged'] or o['reset'] or o['fatal']],
                    'remove_repeated_noops':[a for a,o in zip(seq,obs) if not o['repeated_observed_noop'] or o['reset'] or o['fatal']],
                }
                result = dict(run=name, game_id=gid, level=lv, original_actions=len(seq), trials={})
                for key, trimmed in cases.items():
                    check = trial(start, trimmed, lv) if len(trimmed) < len(seq) else dict(solves=True, completion_action=len(seq), exact_end=True)
                    check['removed_candidates'] = len(seq)-len(trimmed)
                    check['kept_actions'] = trimmed if check['removed_candidates'] else None
                    result['trials'][key] = check
                    # Successful hindsight only; retain original on failure. Count early completion exactly.
                    if check['solves']:
                        n = check['completion_action']
                        counterfactual[key][lv] = n
                        totals[key] += len(seq)-n
                deletions.append(result)
                graph = json.loads((ROOT/row['graph_path']).read_text()) if row['graph_exhausted_with_matching_root'] else None
                distances = []
                if graph:
                    distances.append(graph['dist'].get(hashlib.md5(np.asarray(fr.frame[-1]).tobytes()).hexdigest()))
                for j, a in enumerate(seq):
                    fr = act(g, a)
                    board = np.asarray(fr.frame[-1])
                    eventfile = ROOT/rel/'artifacts'/f'{gid}_p0_events.jsonl'
                    if graph:
                        distances.append(0 if fr.levels_completed > lv else graph['dist'].get(hashlib.md5(board.tobytes()).hexdigest()))
                assert fr.levels_completed == lv+1, (name,gid,lv,fr.levels_completed)
                if graph:
                    oracle_traces.append(dict(run=name,game_id=gid,level=lv,actions=len(seq),baseline=bases[lv],
                                              graph_path=row['graph_path'],d=distances,known_states=sum(x is not None for x in distances),
                                              all_states_known=all(x is not None for x in distances),
                                              root_distance=distances[0],total_action_excess=len(seq)-distances[0]))
                k += apl[lv]
            for x in hist[k:]:
                fr = act(g,x['action'])
            assert fr.levels_completed == done
            assert (fr.state == GameState.WIN) == (r['state'] == 'won')
            replay_checks.append(dict(run=name,game_id=gid,logged_levels=done,replayed_levels=fr.levels_completed,
                                      logged_state=r['state'],engine_state=fr.state.name,actions=len(hist),ok=True))
            for key in scores:
                scores[key].append(game_score(bases,counterfactual[key],done))
        s=dict(run=name, games_replayed=len(records), verified_removed_actions=totals,
               hindsight_deletion_mean_scores={key:sum(v)/len(v) for key,v in scores.items()})
        run_summaries.append(s)
        print(json.dumps(s), flush=True)
    for fname, rr in [('replay-checks.jsonl',replay_checks),('deletion-checks.jsonl',deletions),('oracle-traces.jsonl',oracle_traces)]:
        (HERE/fname).write_text(''.join(json.dumps(r)+'\n' for r in rr))
    (HERE/'replay-summary.json').write_text(json.dumps(run_summaries,indent=2)+'\n')


if __name__ == '__main__':
    main()
