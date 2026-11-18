"""Audit actual request histories and timing brackets, preserving raw logs."""
from pathlib import Path
from collections import Counter
import ast
import base64
import hashlib
import json
import re
import zlib

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent


def sig(message):
    return hashlib.sha256(json.dumps(message,sort_keys=True).encode()).hexdigest()


def text(message):
    content=message.get('content','')
    if isinstance(content,list):content='\n'.join(p.get('text','') for p in content if isinstance(p,dict))
    return str(content)+'\n'+str(message.get('reasoning_content',''))+'\n'+json.dumps(message.get('tool_calls') or [])


audit={'runs':{},'source_identity':{}}
for label in ['b1-r1','b1-r2']:
    p=ROOT/'solvers/m2_b1/experiment/pair-1'/label
    nb=next((p/'kernel').glob('*.ipynb'))
    cell=next(''.join(c['source']) for c in json.loads(nb.read_text())['cells'] if 'M2 B1 installed' in ''.join(c['source']))
    tree=ast.parse(cell)
    packed=next(node.args[0].value for node in ast.walk(tree) if isinstance(node,ast.Call)
                and isinstance(node.func,ast.Attribute) and node.func.attr=='b64decode'
                and node.args and isinstance(node.args[0],ast.Constant))
    files=json.loads(zlib.decompress(base64.b64decode(packed)))
    source=files['inference/agent/tool_agent.py']['source']
    assert hashlib.sha256(source.encode()).hexdigest()==files['inference/agent/tool_agent.py']['after']
    local=ROOT/'solvers/m2_b1/src/inference/agent/tool_agent.py'
    audit['source_identity'][label]={'packaged_sha256':hashlib.sha256(source.encode()).hexdigest(),
                                   'local_matches_packaged':source==local.read_text()}
    assert source==local.read_text()
    for gid in ['cn04-2fe56bfb','s5i5-18d95033','tn36-ef4dde99']:
        requests=[]
        log=p/'output'/f'{gid}_p0_requests.jsonl'
        for line_number,line in enumerate(log.open(),1):
            req=json.loads(line)
            if req.get('event')!='request':continue
            states=[(m,re.search(r'Current state: step (\d+), level (\d+)\.',text(m)))
                    for m in req['messages'] if m['role']=='user']
            states=[(m,s) for m,s in states if s]
            requests.append({'line':line_number,'analysis_step':req['analysis_step'],'action':req['action'],
                             'messages':req['messages'],'states':[(int(s[1]),int(s[2])) for m,s in states]})
        first_l2=next(r for r in requests if r['states'] and r['states'][-1][1]==2)
        opening_step=first_l2['states'][-1][0]
        opening_message=next(m for m in first_l2['messages'] if m['role']=='user'
                             and re.search(rf'Current state: step {opening_step}, level 2\.',text(m)))
        opening_signature=sig(opening_message)
        drops=[]
        for previous,current in zip(requests,requests[1:]):
            n0,n1=len(previous['messages']),len(current['messages'])
            if n1<n0-4:
                old=Counter(map(sig,previous['messages']));new=Counter(map(sig,current['messages']))
                removed=list((old-new).elements())
                drops.append({'source_line':current['line'],'analysis_step':current['analysis_step'],
                              'action':current['action'],'messages_before':n0,'messages_after':n1,
                              'removed_message_count':len(removed),'opening_retained':(opening_step,2) in current['states'],
                              'opening_message_unchanged':opening_signature in new,
                              'level_after':current['states'][-1][1] if current['states'] else None})
        l2drops=[d for d in drops if d['level_after']==2]
        counter=json.loads((p/'output/artifacts'/f'{gid}_p0_tool_runtime_state_b1.json').read_text())
        audit['runs'][label+'/'+gid[:4]]={'request_log':str(log.relative_to(ROOT)),
                                      'observed_history_reductions_l2':l2drops,'whole_game_history_reductions':drops,
                                      'reported_b1_request_time_trim_counter':counter['trims'],
                                      'opening_step':opening_step,
                                      'request_count':len(requests)}
        if gid.startswith('s5i5') and label=='b1-r1':
            selected=next(r for r in requests if r['line']==73)
            corpus='\n'.join(text(m) for m in selected['messages'])
            coupling='When sky grew up, the chain moved yellow up AND green up, but not orange.'
            location=next((i for i,m in enumerate(selected['messages']) if coupling in text(m)),None)
            assert location is not None
            audit['feedback_checkpoint']={'source_line':73,'messages':len(selected['messages']),
                'user_state_steps':selected['states'],
                'explicit_correct_coupling_statement_present':True,
                'coupling_message_index':location,'coupling_message_role':selected['messages'][location]['role'],
                'coupling_statement':coupling,
                'coupling_mentions':[m.group(0) for m in re.finditer(r'.{0,60}(?:green.{0,25}up|yellow.{0,25}green.{0,25}up).{0,80}',corpus,re.I)][:8],
                'note':'Substring cues are navigation evidence only; raw message membership must be read before a retention claim.'}
        print(label,gid[:4],'counter',counter['trims'],'L2 history reductions',len(l2drops),
              'opening retained',all(d['opening_retained'] for d in l2drops))
(OUT/'request-context-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
