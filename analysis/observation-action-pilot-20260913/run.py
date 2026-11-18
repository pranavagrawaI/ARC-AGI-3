"""Fresh paired consistency trial. Imports frozen replay/transport without editing it."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import time
import urllib.request
from typing import Any

HERE = Path(__file__).resolve().parent
OLD = HERE.parent/'stop-checkpoint-pilot-20260913/pilot.py'
spec = importlib.util.spec_from_file_location('frozen_stop_pilot', OLD)
assert spec is not None and spec.loader is not None
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
PAIRS = [(51, ('baseline','consistency')), (52, ('consistency','baseline')), (53, ('baseline','consistency'))]
COMMON = '''The host supplies exact current-frame color measurements with each request. They are raw geometry, not objects, mechanics, or goals. bbox is [min_row,max_row,min_col,max_col], inclusive over ALL cells of that color, not a moving object's identity. In each decide call, choose one present color relevant to your decision and read back its current count and bbox in readback. planned_actions records the final intended sequence; actions records the submitted sequence. Both may be empty to stop. Keep rationale and prediction concise and consistent with that final sequence. The host does not judge whether a goal, theory, or action is useful.'''
TREATMENT = '''CONSISTENCY GATE IS ACTIVE: Before submitting, verify readback against the CURRENT measurement table (not old notes or the union of old/new diff cells), and ensure planned_actions exactly equals actions. A mismatch executes no action and returns factual feedback. Correct it within the remaining call budget. The host verifies only those two contracts, never your game theory or natural-language rationale.'''
TOOLS = copy.deepcopy(base.TOOLS)
params = TOOLS[0]['function']['parameters']
params['properties']['planned_actions'] = copy.deepcopy(params['properties']['actions'])
params['properties']['readback'] = {'type':'object','properties':{
    'color':{'type':'integer','minimum':0,'maximum':15},
    'count':{'type':'integer','minimum':1,'maximum':4096},
    'bbox':{'type':'array','items':{'type':'integer','minimum':0,'maximum':63},'minItems':4,'maxItems':4}},
    'required':['color','count','bbox'],'additionalProperties':False}
params['required'] += ['planned_actions','readback']


def measure(grid):
    groups = {}
    for r,row in enumerate(grid):
        for c,value in enumerate(row): groups.setdefault(value,[]).append((r,c))
    return [{'color':color,'count':len(cells),'bbox':[min(r for r,c in cells),max(r for r,c in cells),min(c for r,c in cells),max(c for r,c in cells)]}
            for color,cells in sorted(groups.items())]


def body(messages,seed):
    b = base.body_for(messages,seed)
    b['tools'] = TOOLS
    b['provider']['only'] = ['dekallm']
    return b


def parse(payload):
    source = payload['choices'][0]['message']
    calls = source.get('tool_calls') or []
    if len(calls)!=1 or calls[0]['function']['name']!='decide': raise ValueError('one decide call required')
    d = json.loads(calls[0]['function']['arguments'])
    if not isinstance(d,dict) or set(d)!=set(params['required']): raise ValueError('decision fields')
    for k in ('actions','planned_actions'):
        if not isinstance(d[k],list) or len(d[k])>2 or any(type(a)is not str or a not in base.ACTIONS for a in d[k]): raise ValueError('action shape')
    for k in ('rationale','prediction','stop_reason'):
        if type(d[k]) is not str: raise ValueError('text shape')
    rb = d['readback']
    if not isinstance(rb,dict) or set(rb)!={'color','count','bbox'}: raise ValueError('readback shape')
    if type(rb['color'])is not int or not 0<=rb['color']<=15 or type(rb['count'])is not int or not 1<=rb['count']<=4096: raise ValueError('readback numbers')
    if not isinstance(rb['bbox'],list) or len(rb['bbox'])!=4 or any(type(x)is not int or not 0<=x<=63 for x in rb['bbox']): raise ValueError('bbox shape')
    # Strip response-only fields, including tool-call index, from the next request.
    message = {'role':'assistant','content':source.get('content'), 'tool_calls':[{'id':calls[0]['id'],'type':'function','function':copy.deepcopy(calls[0]['function'])}]}
    return message,calls[0]['id'],d


def validate(d,grid):
    actual = next((m for m in measure(grid) if m['color']==d['readback']['color']),None)
    return {'readback_ok':d['readback']==actual, 'actions_agree':d['actions']==d['planned_actions'], 'actual_readback':actual}


def play(transport,transitions,packet,name,seed,out=HERE):
    from arcengine import GameAction
    env,raw = base.environment(transitions)
    start_level = raw.levels_completed
    messages = [{'role':'system','content':base.SYSTEM+'\n'+COMMON},{'role':'user','content':json.dumps(packet)}]
    if name=='consistency': messages.append({'role':'user','content':TREATMENT})
    summary: dict[str,Any] = dict(seed=seed,arm=name,calls=0,actions=0,level_gain=0,decisions=0,readback_failures=0,action_mismatches=0,rejections=0,clean_accepted=0,repaired_executed=0,stop='call_cap')
    previous_rejection = False
    for turn in range(4):
        messages.append({'role':'user','content':json.dumps({'new_actions_remaining':6-summary['actions'],'calls_remaining':4-turn,'current_color_measurements':measure(raw.frame[-1].tolist())})})
        payload = transport.call(body(messages,seed),f'{seed}-{name}-{turn}')
        summary['calls'] += 1
        if payload.get('provider')!='DekaLLM': raise ValueError('unexpected provider')
        try: message,call_id,d = parse(payload)
        except (ValueError,KeyError,TypeError,IndexError) as e:
            summary['stop']='invalid_decision'
            base.append(out/'trajectories.jsonl',dict(seed=seed,arm=name,turn=turn,event='invalid_decision',error_type=type(e).__name__))
            break
        verdict = validate(d,raw.frame[-1].tolist())
        passed = verdict['readback_ok'] and verdict['actions_agree']
        reject = name=='consistency' and not passed
        summary['decisions']+=1
        summary['readback_failures']+=not verdict['readback_ok']
        summary['action_mismatches']+=not verdict['actions_agree']
        summary['rejections']+=reject
        summary['clean_accepted']+=passed
        base.append(out/'trajectories.jsonl',dict(seed=seed,arm=name,turn=turn,event='decision',decision=d,validation=verdict,rejected=reject))
        messages.append(message)
        if reject:
            messages.append({'role':'tool','tool_call_id':call_id,'content':json.dumps({'executed_actions':[],'consistency_failure':verdict,'instruction':'Correct the factual readback or the declared/submitted sequence. No game mechanic or action choice is supplied.'})})
            previous_rejection=True
            continue
        if not d['actions']:
            summary['stop']='voluntary_stop'
            break
        if previous_rejection: summary['repaired_executed']+=1
        previous_rejection=False
        observed=[]
        for action in d['actions'][:6-summary['actions']]:
            before=raw.frame[-1].tolist()
            old_level=raw.levels_completed
            raw=env.step(GameAction[base.ACTIONS[action]])
            assert raw is not None
            summary['actions']+=1
            summary['level_gain']=raw.levels_completed-start_level
            row=dict(action=action,before=before,after=raw.frame[-1].tolist(),state=raw.state.name,levels_completed=raw.levels_completed)
            base.append(out/'trajectories.jsonl',dict(seed=seed,arm=name,turn=turn,event='action',**row))
            observed.append(dict(action=action,changes=base.changes(before,row['after']),state=raw.state.name,levels_completed=raw.levels_completed))
            if raw.levels_completed!=old_level or raw.state.name in ('WIN','GAME_OVER'): break
        messages.append({'role':'tool','tool_call_id':call_id,'content':json.dumps(observed)})
        if raw.state.name in ('WIN','GAME_OVER'): summary['stop']=raw.state.name;break
        if summary['actions']==6: summary['stop']='action_cap';break
    return summary


def files():
    return [HERE/'run.py',HERE/'test_run.py',HERE/'audit.py',HERE/'EXPERIMENT-CARD.md',HERE/'ENDPOINTS.json',*base.subjects()]


def prepare():
    if (HERE/'STARTED.json').exists(): raise RuntimeError('frozen run')
    with urllib.request.urlopen('https://openrouter.ai/api/v1/models/qwen/qwen3.8-27b/endpoints',timeout=30) as response:
        metadata=json.load(response)
    base.dump(HERE/'ENDPOINTS.json',metadata)
    endpoint=next(e for e in metadata['data']['endpoints'] if e['tag']=='dekallm')
    assert endpoint['status']==0 and endpoint['supports_tool_choice']['function']
    assert {'reasoning','temperature','seed','max_tokens','tools','tool_choice'}<=set(endpoint['supported_parameters'])
    transitions,packet=base.evidence()
    _,first=base.environment(transitions);_,second=base.environment(transitions)
    assert first.frame[-1].tolist()==second.frame[-1].tolist()
    sample=body([{'role':'system','content':base.SYSTEM+'\n'+COMMON},{'role':'user','content':json.dumps(packet)},{'role':'user','content':TREATMENT},{'role':'user','content':json.dumps(measure(first.frame[-1].tolist()))}],51)
    assert len(json.dumps(sample,separators=(',',':')).encode())<=24000
    base.dump(HERE/'PRELAUNCH.json',{'passed':True,'files':{str(p):base.sha(p) for p in files()},'tests_sha256':base.sha(HERE/'TESTS.txt'),'replayed_prefix':7,'matching_replays':2})
    print('Preflight passed: replay, endpoint compatibility, request size and source freeze.')


def credit(key,label):
    request=urllib.request.Request('https://openrouter.ai/api/v1/credits',headers={'Authorization':'Bearer '+key})
    with urllib.request.urlopen(request,timeout=30) as response: data=json.load(response)['data']
    enough=data['total_credits']-data['total_usage']>=1
    base.dump(HERE/f'QUOTA-{label}.json',{'available_at_least_one_dollar':enough})
    return enough


def launch():
    if (HERE/'STARTED.json').exists() or (HERE/'raw').exists(): raise RuntimeError('one shot only')
    frozen=json.loads((HERE/'PRELAUNCH.json').read_text())
    assert frozen['passed'] and all(base.sha(Path(p))==h for p,h in frozen['files'].items())
    assert base.sha(HERE/'TESTS.txt')==frozen['tests_sha256']
    key=None
    with Path('/home/pranav/projects/arc-agi-3/.env').open() as f:
        for line in f:
            name,sep,value=line.strip().removeprefix('export ').partition('=')
            if sep and name.strip()=='OPENROUTER_API_KEY':
                if key is not None: raise RuntimeError('duplicate authorized key')
                value=value.strip();key=ast.literal_eval(value) if value.startswith(('"',"'")) else value
    if not isinstance(key,str) or not key: raise RuntimeError('authorized key unavailable')
    base.dump(HERE/'STARTED.json',{'time_unix':time.time(),'prelaunch_sha256':base.sha(HERE/'PRELAUNCH.json'),'files':frozen['files']})
    transport=base.Transport(key,HERE)
    result={'classification':'COMPLETE_PILOT','arms':[]}
    try:
        if not credit(key,'before'): raise RuntimeError('insufficient credit')
        transitions,packet=base.evidence()
        for seed,order in PAIRS:
            for name in order:
                result['arms'].append(play(transport,transitions,packet,name,seed))
                base.dump(HERE/'PROGRESS.json',result)
        credit(key,'after')
    except Exception as e:
        result.update(classification='INTEGRATION_INVALID',error_type=type(e).__name__,error=str(e))
    ledger=HERE/'attempts.jsonl'
    rows=[json.loads(l) for l in ledger.read_text().splitlines()] if ledger.exists() else []
    unknown=any(r['status']=='failed' for r in rows) or sum(r['status']=='started' for r in rows)!=sum(r['status']=='completed' for r in rows)
    result.update(request_attempts=transport.index,receipted_cost=transport.spent,total_cost=None if unknown else transport.spent,wall_seconds=time.monotonic()-transport.started)
    base.dump(HERE/'RESULT.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    import sys
    {'prepare':prepare,'launch':launch}[sys.argv[1]]()
