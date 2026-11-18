"""Replay generated predictors on observed transitions, with explicit coverage.

Functions are extracted unchanged. Pure predictors see recorded predecessor
states, not engine internals. No model inference or real game interaction.
"""
from pathlib import Path
import ast
import copy
from collections import deque
import json
import re

OUT = Path(__file__).resolve().parent
R = json.loads((OUT/'records.json').read_text())
E = json.loads((OUT/'engine-replay.json').read_text())


def functions_at(label,line,ns):
    f=next(f for b in R[label]['blocks'] for f in b['fragments'] if f['source_line']==line)
    code=re.search(r'<parameter=code>\s*(.*?)\s*</parameter>',f['text'],re.S)[1]
    tree=ast.parse(code)
    only=ast.Module(body=[node for node in tree.body if isinstance(node,ast.FunctionDef)],type_ignores=[])
    exec(compile(only,f'recorded-{label}-{line}','exec'),ns)


bad={}
functions_at('b1-r1',4731,bad)
functions_at('b1-r1',7835,bad)
good={'OBS':{(r,c) for r,row in enumerate(R['control-pc']['entry_board']) for c,col in enumerate(row)
             if col==15 or (r>=52 and col!=5)}}
functions_at('control-pc',4862,good)
functions_at('control-pc',5635,good)
COLORS='OSYN'
PANELS={'O':3,'S':18,'Y':33,'N':48}
report={'sources':{'failed_predictor':{'run':'b1-r1','line':7835,'depends_on_line':4731},
                   'successful_predictor':{'run':'control-pc','lines':[4862,5635]}},
        'transition_tests':{},'scope':'Retrospective local predictor checks, not intervention benefit or live model repair.'}


def good_predict(state,key,delta):
    lengths=tuple(state[k]['L'] for k in COLORS)
    candidate=list(lengths);candidate[COLORS.index(key)]+=delta;candidate=tuple(candidate)
    if not good['ok3'](candidate):return None
    Lo,Ls,Ly,Lg=candidate
    return {'O':{'axis':'H','r':39,'c':10,'L':Lo},
            'S':{'axis':'V','dir':-1,'r':41-Ls,'c':10+Lo,'L':Ls},
            'Y':{'axis':'H','r':38-Ls,'c':11+Lo,'L':Ly},
            'N':{'axis':'V','dir':1,'r':39-Ls,'c':11+Lo+Ly,'L':Lg}}


for label,rec in R.items():
    actual=E['runs'][label]['states']
    assert len(actual)==len(rec['actions'])+1
    rows=[]
    for action,pre,post in zip(rec['actions'],actual,actual[1:]):
        a=action['action']
        xy=a.get('data',{})
        # Interpret the physical panel hit, including nearby click coordinates.
        # The source uses the 13-pixel panel midpoint: left retracts, right grows.
        if a['id']!='ACTION6' or not 54<=xy.get('y',-1)<=60:
            continue
        hits=[(key,xy['x']-left) for key,left in PANELS.items() if left<=xy.get('x',-1)<left+13]
        if not hits or hits[0][1]==6:continue
        key,offset=hits[0];delta=3 if offset>6 else -3
        assert post['state'][key]['L']-pre['state'][key]['L'] in [0,delta],(label,action['local_action'])
        predictions={'failed_push_model':bad['growS'](copy.deepcopy(pre['state']),bad['static_set'](),key)
                     if delta>0 else bad['shrinkS'](copy.deepcopy(pre['state']),key),
                     'successful_chain_model':good_predict(pre['state'],key,delta)}
        outcomes={}
        for name,pred in predictions.items():
            if pred is None:
                status='REJECTS_OBSERVED_CHANGE' if pre['state']!=post['state'] else 'NO_SUCCESSOR_ON_NOOP'
            else:
                status='MATCH' if pred==post['state'] else 'MISMATCH'
            outcomes[name]={'status':status,'prediction':pred}
        rows.append({'local_action':action['local_action'],'minutes_since_entry':action['minutes_since_entry'],
                     'interpreted_panel_action':[key,delta],
                     'action':action['action'],'input_state':pre['state'],'actual_successor':post['state'],
                     'outcomes':outcomes})
    assert rows,label
    counts={}
    for name in ['failed_push_model','successful_chain_model']:
        statuses=[row['outcomes'][name]['status'] for row in rows]
        counts[name]={s:statuses.count(s) for s in sorted(set(statuses))}
    report['transition_tests'][label]={'checked':len(rows),'predictor_calls_per_model':len(rows),
                                      'successor_comparisons':{name:sum(row['outcomes'][name]['prediction'] is not None for row in rows)
                                                               for name in predictions},
                                      'counts':counts,'rows':rows}
    print(label,'coverage',len(rows),'counts',counts)

# Causally available subset for the bad model: r1 actions already seen before
# its definition, at global action 42 (local 22). Do not include later evidence.
past=[row for row in report['transition_tests']['b1-r1']['rows'] if row['local_action']<=22]
violations=[row for row in past if row['outcomes']['failed_push_model']['status'] in ['MISMATCH','REJECTS_OBSERVED_CHANGE']]
assert violations
report['available_when_failed_predictor_was_written']={
    'checked':len(past),'predictor_calls':len(past),
    'successor_comparisons':sum(row['outcomes']['failed_push_model']['prediction'] is not None for row in past),
    'contradictions':len(violations),'first_contradiction_local_action':violations[0]['local_action'],
    'row':violations[0],'note':'All these transitions precede definition line 7835; this packet uses no future frames.'}


def find_plan(observed_state):
    start=tuple(observed_state[k]['L'] for k in COLORS)
    q=deque([start]);parents={start:None};edge={}
    while q:
        state=q.popleft()
        if good['marker'](state)==(31,52):
            path=[];cur=state
            while parents[cur] is not None:
                path.append(edge[cur]);cur=parents[cur]
            return {'start':start,'goal':state,'explored':len(parents),'tokens':list(reversed(path))}
        for i,key in enumerate(COLORS):
            for delta in (3,-3):
                candidate=list(state);candidate[i]+=delta;candidate=tuple(candidate)
                if candidate in parents or not good['ok3'](candidate):continue
                parents[candidate]=state;edge[candidate]=key+('g' if delta>0 else 's');q.append(candidate)
        assert len(parents)<100000,'bounded search exceeded'
    return None


report['offline_plans']={}
for label,stop in [('b1-r1',22),('b1-r2',46)]:
    state=E['runs'][label]['states'][stop]['state']
    plan=find_plan(state)
    assert plan,label
    plan['after_local_action']=stop
    report['offline_plans'][label]=plan
    print('offline plan',label,'after',stop,'clicks',len(plan['tokens']),'states',plan['explored'])
(OUT/'model-check.json').write_text(json.dumps(report,indent=2)+'\n')
print('already observed contradiction:',violations[0]['local_action'],violations[0]['outcomes']['failed_push_model']['status'])
