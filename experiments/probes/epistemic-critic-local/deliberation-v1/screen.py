"""Repeatable deliberation diagnostic; never executes Actor code."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.request

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from claim_screen import decode_json
from paired_commitment import validate_critique
ROOT=HERE.parents[3]
OUT=ROOT/'analysis/local-probes/20260911-critic-deliberation-v1'

def write(name,x): (OUT/name).write_text(json.dumps(x,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def prepare():
    OUT.mkdir(exist_ok=False)
    prompt=(HERE.parent/'framework-v1/system.md').read_text()
    brief='''\nFor this diagnostic, replace the top-level response schema with:
{"commitment":"brief statement of the Actor commitment being reviewed",
 "grounds":"brief statement of the supplied evidence offered for that commitment",
 "critique":{the original five fields}}.
Give commitment and grounds first, at most 50 words together. These are concise
review statements, not a private reasoning transcript. Keep the original critique
rules and word limits. This diagnostic wrapper applies even for no_concern.'''
    pairs=[('scope',['One click on a gray square in state A produced no visible change. No other click targets or states were tested.'],
        'That click did nothing, so clicking cannot interact with this game. I will remove CLICK from all future candidate actions.',
        'The gray-square click in state A produced no visible change. Other targets remain untested. I will keep CLICK available and take one movement probe now.',
        'Reject extrapolation from one target/state to an entire action class. Preserve untested options without asserting clicks work or requiring all targets to be tested.'),
        ('agreement',['A progress-meter explanation and an action-counter explanation both predict the same one-cell strip change after RIGHT. RIGHT produced that strip change. No level completed.'],
        'The strip changed exactly as my progress-meter explanation predicted. This eliminates the action-counter explanation. I will optimize strip growth as the objective.',
        'The strip change matches both explanations, so it does not distinguish their roles. I will leave the objective unresolved and take one bounded interaction probe.',
        'Reject selective elimination using a prediction shared by both explanations. Do not assert which objective is true; allow bounded inquiry under uncertainty.')]
    cases=[]
    for pair,obs,bad,good,rubric in pairs:
        for kind,actor in [('fault',bad),('sound',good)]:
            cases.append({'id':pair+'-'+kind,'observations':obs,'actor_reasoning':actor,
                          'expected':'concern' if kind=='fault' else 'no_concern','rubric':rubric})
    write('CASES.json',cases)
    template=json.loads(next((ROOT/'analysis/local-probes/20260911-critic-resource-screen-v1').glob('*.request.json')).read_text())
    order=[]
    for repeat in range(2):
        for i,c in enumerate(cases):
            arms=['thinking','direct','explicit'];shift=(i+repeat)%3;arms=arms[shift:]+arms[:shift]
            for arm in arms:
                name=f'{len(order):02d}-{c["id"]}-r{repeat}-{arm}'
                body=dict(template);body.update(max_tokens=2048,reasoning={'enabled':arm=='thinking'},
                   chat_template_kwargs={'enable_thinking':arm=='thinking'},messages=[
                   {'role':'system','content':prompt+(brief if arm=='explicit' else '')},
                   {'role':'user','content':json.dumps({k:c[k] for k in ('observations','actor_reasoning')})}])
                write(name+'.request.json',body);order.append({'name':name,'arm':arm,'case':c['id'],'repeat':repeat})
    write('ORDER.json',order)
    reserve=sum((len((OUT/(r['name']+'.request.json')).read_bytes())+4096)*.42/1e6+2048*3/1e6 for r in order)
    assert reserve<.25,reserve
    (OUT/'EXPERIMENT-CARD.md').write_text('''# Deliberation diagnostic v1
Question: Does bounded thinking or an explicit review-target statement improve epistemic judgment over direct JSON?
Contrast: Direct thinking-off / thinking-on / thinking-off plus commitment and grounds wrapper. Same framework prompt otherwise, same evidence and 2048 total completion ceiling. Four new authored faulty/sound cases, two samples per arm, cyclic order. The direct ceiling differs from earlier 384-token runs; comparisons are within this card.
Posture: Diagnostic, not gameplay efficacy.
Prediction: Thinking or explicit identification increases grounded fault detection and valid revisions without false objections on sound controls.
Measure: Final critique assessment, quote binding, supported gap and revision, sound-control restraint, repeat stability, delivery compliance, latency, total/reasoning tokens. Grade final output only; do not use private reasoning text as substitute for a final critique. Preserve raw provider response as evidence.
Cap: 24 calls, $0.25 conservative reservation, 900-second admission window, 45 seconds per request, 2048 total generated tokens each, zero environment actions, same Qwen3.8-27B/CoreWeave FP8 route, no fallback or retries.
Stop: Complete calls or abort on transport/provider/accounting failure. If a thinking-on response reports no positive reasoning tokens, or a thinking-off response reports reasoning, stop: mode comparison unresolved. Truncation is an output failure and does not trigger retry; continue to measure bounded-mode feasibility.
Invalid if: Input drift, evaluator labels leaked, wrong model/provider, missing costs or requests. Truncated/malformed/overlong output never becomes no_concern. Reasoning-budget exhaustion limits conclusions to this bounded configuration.
Result map: Better grounded stable responses with acceptable cost -> fresh full-context/Actor uptake test; no improvement -> no promotion; mode unsupported -> report unavailable comparison; truncation -> report bounded-mode infeasibility, no silent budget increase. Single pairs and two repeats cannot establish general reliability.
Outputs: This directory: frozen prompts/requests/cases/order, raw responses and accounting, mechanical audit and manual grades.
API reference: https://openrouter.ai/docs/guides/best-practices/reasoning-tokens ; actual response accounting, not requested flags alone, establishes mode exposure.
''')
    sources=[Path(__file__),HERE.parent/'framework-v1/system.md',HERE.parent/'claim_screen.py',HERE.parent/'paired_commitment.py',HERE.parent/'reasoning_screen.py']
    write('FROZEN.json',{'reserved_usd':reserve,'files':{str(p):sha(p) for p in list(OUT.iterdir())+sources if p.is_file()}})
    print(json.dumps({'prepared':len(order),'reservation':reserve}))

def execute():
    frozen=json.loads((OUT/'FROZEN.json').read_text())
    assert all(sha(Path(p))==h for p,h in frozen['files'].items())
    key=os.environ.get('OPENROUTER_API_KEY')
    for p in [ROOT/'.env',Path('/home/pranav/projects/arc-agi-3/.env')]:
        if key:break
        if p.exists():
            for line in p.read_text().splitlines():
                if line.startswith('OPENROUTER_API_KEY='):key=line.split('=',1)[1].strip().strip('\"\'')
    assert key
    with (OUT/'STARTED.json').open('x') as f:json.dump({'time':time.time()},f)
    result={'status':'running','calls':[],'cost_usd':0.};start=time.monotonic()
    try:
        for item in json.loads((OUT/'ORDER.json').read_text()):
            assert time.monotonic()-start<900
            name=item['name'];raw=(OUT/(name+'.request.json')).read_bytes();body=json.loads(raw)
            assert body['model']=='qwen/qwen3.8-27b' and body['provider']['only']==['coreweave/fp8'] and body['provider']['allow_fallbacks'] is False
            row={**item,'status':'attempted'};result['calls'].append(row);write('EXECUTION.json',result)
            req=urllib.request.Request('https://openrouter.ai/api/v1/chat/completions',raw,{'Content-Type':'application/json','Authorization':'Bearer '+key})
            t=time.monotonic()
            with urllib.request.urlopen(req,timeout=45) as reply:answer=reply.read()
            (OUT/(name+'.response.json')).write_bytes(answer)
            r=json.loads(answer);choice=r['choices'][0];msg=choice['message'];usage=r['usage']
            row.update(seconds=time.monotonic()-t,usage=usage,provider=r.get('provider'),finish_reason=choice.get('finish_reason'))
            assert r.get('provider')=='CoreWeave'
            assert isinstance(usage.get('cost'),(int,float));result['cost_usd']+=usage['cost'];assert result['cost_usd']<.25
            rt=usage.get('completion_tokens_details',{}).get('reasoning_tokens',0)
            assert (rt>0) if item['arm']=='thinking' else (rt==0 and not(msg.get('reasoning') or msg.get('reasoning_content'))),'reasoning mode not verified'
            try:
                assert choice['finish_reason']=='stop','truncated or unfinished output'
                value=decode_json(msg.get('content') or '')
                if item['arm']=='explicit':
                    assert set(value)=={'commitment','grounds','critique'}
                    row['review_statement']={k:value[k] for k in ('commitment','grounds')}
                    assert all(isinstance(value[k],str) for k in ('commitment','grounds'))
                    row['wrapper_within_budget']=sum(len(value[k].split()) for k in ('commitment','grounds'))<=50
                    value=value['critique']
                row['parsed_critique']=value
                critique=validate_critique(value)
                actor=json.loads(body['messages'][1]['content'])['actor_reasoning']
                assert not critique['actor_quote'] or critique['actor_quote'] in actor,'quote not in Actor material'
                if critique['assessment']=='no_concern':assert all(not v for k,v in critique.items() if k!='assessment')
                assert row.get('wrapper_within_budget',True),'wrapper word budget'
                row['status']='valid'
            except Exception as e:row.update(status='output_failure',error=str(e))
            write('EXECUTION.json',result);print(name,row['status'],'reasoning_tokens',rt,flush=True)
        result['status']='completed'
    except Exception as e:result.update(status='aborted',error=f'{type(e).__name__}: {e}')
    write('EXECUTION.json',result);print(json.dumps({'status':result['status'],'cost':result['cost_usd']}))

if __name__=='__main__':{'prepare':prepare,'execute':execute}[sys.argv[1]]()
