"""Post-run audit of raw requests, receipts, treatment and real action replay. No API calls."""
import collections
import json
from pathlib import Path
import pilot
from arcengine import GameAction

out = Path(__file__).resolve().parent
result = json.loads((out/'RESULT.json').read_text())
started = json.loads((out/'STARTED.json').read_text())
checks = {'complete':result['classification']=='COMPLETE_PILOT',
          'source_unchanged':all(pilot.sha(Path(p))==h for p,h in started['source_sha256'].items()),
          'prelaunch_unchanged':pilot.sha(out/'PRELAUNCH.json')==started['prelaunch_sha256'],
          'wall_cap':result['wall_seconds']<=900}
events = [json.loads(l) for l in (out/'trajectories.jsonl').read_text().splitlines()]
ledger = [json.loads(l) for l in (out/'attempts.jsonl').read_text().splitlines()]
requests = [x for x in ledger if x['status']=='started']
ends = [x for x in ledger if x['status']=='completed']
checks['all_attempts_completed'] = len(requests)==len(ends)==24 and len(ledger)==48
checks['unique_arm_identities'] = len({(r['seed'],r['arm']) for r in result['arms']})==6
base_context = {}
aggregate = collections.defaultdict(lambda:dict(actions=0,calls=0,level_gain=0,voluntary_stops=0))
by_request = {}
for row in requests:
    body = json.loads((out/'raw'/f"{row['index']:02d}-request.json").read_text())
    seed, arm, turn = row['label'].split('-')
    by_request[(int(seed),arm,int(turn))] = body
    treatment = sum(m.get('content')==pilot.CHECKPOINT for m in body['messages'])
    checks[f"treatment_{row['index']}"] = treatment==(1 if arm=='checkpoint' else 0)
    checks[f"request_cap_{row['index']}"] = len(json.dumps(body,separators=(',',':')).encode())<=24000
    checks[f"sampling_{row['index']}"] = body['seed']==int(seed) and body['temperature']==.7
    if turn=='0':
        base_context[(seed,arm)] = [m for m in body['messages'] if m.get('content')!=pilot.CHECKPOINT]
for seed in ('41','42','43'):
    checks[f'matched_initial_context_{seed}'] = base_context[(seed,'baseline')]==base_context[(seed,'checkpoint')]
transitions, _ = pilot.evidence()
for summary in result['arms']:
    seed,name = summary['seed'],summary['arm']
    relevant = [x for x in events if x['seed']==seed and x['arm']==name]
    decisions = [x for x in relevant if x['event']=='decision']
    actions = [x for x in relevant if x['event']=='action']
    env, raw = pilot.environment(transitions)
    level = raw.levels_completed
    for row in actions:
        assert raw.frame[-1].tolist()==row['before']
        raw = env.step(GameAction[pilot.ACTIONS[row['action']]])
        assert raw is not None
        assert raw.frame[-1].tolist()==row['after']
        assert raw.levels_completed==row['levels_completed'] and raw.state.name==row['state']
    checks[f'{seed}_{name}_action_replay'] = True
    checks[f'{seed}_{name}_counts'] = len(actions)==summary['actions']<=6 and len(decisions)==summary['calls']<=4
    checks[f'{seed}_{name}_level_gain'] = raw.levels_completed-level==summary['level_gain']
    checks[f'{seed}_{name}_decision_actions'] = [a for d in decisions for a in d['decision']['actions']]==[a['action'] for a in actions]
    checks[f'{seed}_{name}_stop'] = (not decisions[-1]['decision']['actions'])==(summary['stop']=='voluntary_stop')
    a = aggregate[name]
    for k in ('actions','calls','level_gain'): a[k]+=summary[k]
    a['voluntary_stops']+=summary['stop']=='voluntary_stop'
accounting = pilot.audit(out)
checks['accounting'] = accounting['passed'] and accounting['performance_valid']
checks['cost_reconciles'] = accounting['total_cost']==result['total_cost']
checks['latency_cap'] = all(r['latency_seconds']<=60 for r in ends)
metrics = {'aggregate':dict(aggregate), 'prompt_tokens':sum(r['usage']['prompt_tokens'] for r in ends),
           'completion_tokens':sum(r['usage']['completion_tokens'] for r in ends),
           'cost':result['total_cost'], 'wall_seconds':result['wall_seconds']}
pilot.dump(out/'FINAL-AUDIT.json', {'passed':all(checks.values()), 'checks':checks, 'metrics':metrics,
                                 'scope':'Observed run integrity and local contrast only, not general competence.'})
print(json.dumps({'passed':all(checks.values()),'checks':len(checks),'metrics':metrics},indent=2))
if not all(checks.values()): raise SystemExit(1)
