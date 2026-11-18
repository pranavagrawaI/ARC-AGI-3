"""Audit receipts and independently replay all recorded environment actions. No network."""
import collections
import json
from pathlib import Path
import run


def audit(out=run.HERE, full=True):
    b=run.base
    accounting=b.audit(out)
    checks={'accounting':accounting['passed']}
    ledger=out/'attempts.jsonl'
    rows=[json.loads(l) for l in ledger.read_text().splitlines()] if ledger.exists() else []
    starts=[r for r in rows if r['status']=='started']
    ends=[r for r in rows if r['status']=='completed']
    wire={}
    for r in starts:
        req=json.loads((out/'raw'/f"{r['index']:02d}-request.json").read_text())
        wire[r['label']]=req
        checks[f"pinned_{r['index']}"]=req['provider'].get('only')==['dekallm'] and req['tools']==run.TOOLS
    for r in ends:
        resp=json.loads((out/'raw'/f"{r['index']:02d}-response.json").read_text())
        checks[f"provider_{r['index']}"]=resp.get('provider')=='DekaLLM'
        checks[f"latency_{r['index']}"]=r['latency_seconds']<=60
    metrics={'attempts':len(starts),'completed':len(ends),'receipted_cost':accounting['receipted_cost'],'total_cost':accounting['total_cost']}
    if full:
        from arcengine import GameAction
        frozen=json.loads((out/'STARTED.json').read_text())
        checks['source_unchanged']=all(b.sha(Path(p))==h for p,h in frozen['files'].items())
        checks['prelaunch_unchanged']=b.sha(out/'PRELAUNCH.json')==frozen['prelaunch_sha256']
        result=json.loads((out/'RESULT.json').read_text())
        checks['complete']=result['classification']=='COMPLETE_PILOT' and len(result['arms'])==6
        checks['wall_cap']=result['wall_seconds']<=900
        checks['cost_reconciliation']=result['total_cost']==accounting['total_cost']
        if checks['complete']:
            events=[json.loads(l) for l in (out/'trajectories.jsonl').read_text().splitlines()]
            aggregate=collections.defaultdict(collections.Counter)
            initial={}
            transitions,packet=b.evidence()
            for s in result['arms']:
                seed,name=s['seed'],s['arm']
                relevant=[r for r in events if r['seed']==seed and r['arm']==name]
                env,raw=b.environment(transitions)
                level=raw.levels_completed
                counts=collections.Counter()
                last_rejected=False
                for turn in range(s['calls']):
                    req=wire[f'{seed}-{name}-{turn}']
                    messages=req['messages']
                    checks[f'{seed}_{name}_{turn}_treatment']=sum(m.get('content')==run.TREATMENT for m in messages)==(name=='consistency')
                    checks[f'{seed}_{name}_{turn}_old_checkpoint_absent']=all(m.get('content')!=b.CHECKPOINT for m in messages)
                    telemetry=json.loads(messages[-1]['content'])
                    checks[f'{seed}_{name}_{turn}_current_measurements']=telemetry['current_color_measurements']==run.measure(raw.frame[-1].tolist())
                    checks[f'{seed}_{name}_{turn}_budget']=telemetry['new_actions_remaining']==6-counts['actions'] and telemetry['calls_remaining']==4-turn
                    if turn==0: initial[(seed,name)]=[m for m in messages if m.get('content')!=run.TREATMENT]
                    ds=[r for r in relevant if r['turn']==turn and r['event']=='decision']
                    actions=[r for r in relevant if r['turn']==turn and r['event']=='action']
                    if not ds:
                        checks[f'{seed}_{name}_{turn}_invalid']=s['stop']=='invalid_decision' and not actions
                        continue
                    d=ds[0]['decision'];v=run.validate(d,raw.frame[-1].tolist());passed=v['readback_ok'] and v['actions_agree'];rejected=name=='consistency' and not passed
                    checks[f'{seed}_{name}_{turn}_validation']=ds[0]['validation']==v and ds[0]['rejected']==rejected
                    counts['decisions']+=1;counts['readback_failures']+=not v['readback_ok'];counts['action_mismatches']+=not v['actions_agree'];counts['rejections']+=rejected;counts['clean_accepted']+=passed
                    if rejected:
                        checks[f'{seed}_{name}_{turn}_no_side_effect']=not actions
                        last_rejected=True
                        continue
                    if last_rejected and actions: counts['repaired_executed']+=1
                    last_rejected=False
                    checks[f'{seed}_{name}_{turn}_action_prefix']=[a['action'] for a in actions]==d['actions'][:len(actions)]
                    for a in actions:
                        assert raw.frame[-1].tolist()==a['before']
                        raw=env.step(GameAction[b.ACTIONS[a['action']]])
                        assert raw is not None
                        assert raw.frame[-1].tolist()==a['after'] and raw.state.name==a['state'] and raw.levels_completed==a['levels_completed']
                        counts['actions']+=1
                counts['calls']=s['calls'];counts['level_gain']=raw.levels_completed-level
                checks[f'{seed}_{name}_counts']=all(s[k]==counts[k] for k in ('calls','actions','level_gain','decisions','readback_failures','action_mismatches','rejections','clean_accepted','repaired_executed'))
                checks[f'{seed}_{name}_caps']=s['calls']<=4 and s['actions']<=6
                counts['voluntary_stops']=s['stop']=='voluntary_stop'
                aggregate[name].update(counts)
            for seed,_ in run.PAIRS: checks[f'{seed}_initial_match']=initial[(seed,'baseline')]==initial[(seed,'consistency')]
            metrics['aggregate']={k:dict(v) for k,v in aggregate.items()}
    answer={'passed':all(checks.values()),'performance_valid':full and checks.get('complete',False) and all(checks.values()) and accounting['performance_valid'],'checks':checks,'metrics':metrics}
    b.dump(out/'FINAL-AUDIT.json',answer)
    return answer


if __name__=='__main__':
    result=audit()
    print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))
    if not result['passed']: raise SystemExit(1)
