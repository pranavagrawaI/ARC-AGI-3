"""Frozen paired prompt diagnostic. No gameplay, no retries."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.request

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from reasoning_screen import cases
from paired_commitment import CRITIC_PROMPT, validate_critique
from claim_screen import decode_json

ROOT = HERE.parents[3]
OUT = ROOT / 'analysis/local-probes/20260911-critic-framework-screen-v1'

def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + '\n')

def prepare():
    OUT.mkdir(exist_ok=False)
    source = ROOT / 'analysis/local-probes/20260911-critic-endurance-v1'
    captured = next(json.loads(l)['request'] for l in (source/'requests.jsonl').open()
                    if json.loads(l)['call'] == 40)
    evidence = json.loads(captured['messages'][1]['content'])
    specs = [{'id':'captured-r8', 'kind':'unchanged captured critic input', 'evidence':evidence,
              'expected':'concern OR qualified no_concern',
              'rubric':'Concern must identify incompatible grounds for reachability without asserting game truth. No concern is permissible because Actor is checking; certifying four reachable positions is unsupported. Empty no_concern earns restraint, not detection.'}]
    repair = {'observations': ['A search reported four reachable positions.',
        'A subsequent placement check under the same modeled constraints permits an edge beyond that set.'],
        'actor_reasoning':'The search result and the placement check disagree. I cannot yet claim only four positions are reachable. I will inspect why these computations disagree before using the result to rule out routes; neither computation alone establishes real game reachability.'}
    specs.append({'id':'repair-control','kind':'authored control','evidence':repair,
                  'expected':'no_concern','rubric':'Do not demand the correction already underway or require environment actions before a bounded consistency investigation.'})
    for c in cases():
        if c['pair'] in ('objective','scope'):
            specs.append({'id':c['id'],'kind':c['kind'], 'evidence':{
                'stipulated_observations':c['observations'],'actor_reasoning':c['actor_reasoning']},
                'expected':c['expected'],'rubric':c['rubric']})
    prompts = {'old':CRITIC_PROMPT,'new':(HERE/'system.md').read_text()}
    requests=[]
    for i,spec in enumerate(specs):
        for arm in (['old','new'] if i%2==0 else ['new','old']):
            body={k:v for k,v in captured.items() if k not in ('messages','tools','tool_choice')}
            body.update(max_tokens=384, messages=[{'role':'system','content':prompts[arm]},
                {'role':'user','content':json.dumps(spec['evidence'])}])
            name=f'{len(requests):02d}-{spec["id"]}-{arm}'
            write(name+'.request.json',body)
            requests.append({'name':name,'case':spec['id'],'arm':arm})
    write('CASES.json',specs);write('ORDER.json',requests)
    reserve=sum((len((OUT/(r['name']+'.request.json')).read_bytes())+4096)*.42/1e6+384*3/1e6 for r in requests)
    assert reserve < .25, reserve
    (OUT/'EXPERIMENT-CARD.md').write_text('''# Critic framework screen v1
Question: Does the framework prompt improve warranted criticism and restraint on development cases?
Posture: Diagnostic evaluation of a prompt repair.
Contrast: Old critic and new framework critic; identical evidence per case, alternating order; six cases, one sample each per arm.
Prediction: New critic identifies scope/objective overcommitments, permits sound exploration and active repair, and avoids certifying the inconsistent reachability result.
Measure: Raw response fields graded against CASES.json. Detection, unsupported objections, unsupported endorsements, execution validity, latency, and cost separate. Empty no_concern is not detection.
Cap: 12 calls; $0.25 conservative total reservation; 480 seconds overall admission window; 30 seconds per HTTP request; 384 total generated tokens per call; zero environment actions. Same existing Qwen endpoint/provider, thinking off.
Stop: Complete 12 calls; abort on transport, provider, reasoning, cost-accounting, or finish-status failure. Preserve malformed semantic responses and continue without retry. No relaunch.
Invalid if: Frozen inputs drift, diagnosis/later outcomes leak into captured input, wrong model/provider, or unaccounted requests. A malformed response is an output failure, never no_concern.
Result map: Better grounded judgments -> candidate for fresh-context/live adoption test; worse -> retain old deployment and inspect mistakes; mixed -> no promotion, identify exact unresolved boundary. These development cases cannot establish generalization or separate-role efficacy.
Outputs: This directory contains exact requests/responses, source hashes, grading criteria, timing and costs. No Actor continuation is generated.
''')
    write('FROZEN.json',{'reserved_usd':reserve,'files':{str(p):hashlib.sha256(p.read_bytes()).hexdigest()
         for p in list(OUT.iterdir())+[HERE/'system.md',Path(__file__)] if p.is_file()}})
    print(json.dumps({'prepared':len(requests),'reserved_usd':reserve}))

def execute():
    frozen=json.loads((OUT/'FROZEN.json').read_text())
    for p,h in frozen['files'].items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    key=os.environ.get('OPENROUTER_API_KEY')
    for p in (ROOT/'.env',Path('/home/pranav/projects/arc-agi-3/.env')):
        if key: break
        if p.exists():
            for line in p.read_text().splitlines():
                if line.startswith('OPENROUTER_API_KEY='): key=line.split('=',1)[1].strip().strip('\"\'')
    assert key,'No credential'
    with (OUT/'STARTED.json').open('x') as f:json.dump({'time':time.time()},f)
    result={'status':'running','calls':[],'cost_usd':0.0};start=time.monotonic()
    try:
        for item in json.loads((OUT/'ORDER.json').read_text()):
            assert time.monotonic()-start<480
            name=item['name'];raw=(OUT/(name+'.request.json')).read_bytes();body=json.loads(raw)
            assert body['model']=='qwen/qwen3.8-27b'
            assert body['provider']['only']==['coreweave/fp8'] and not body['provider']['allow_fallbacks']
            assert body['reasoning']['enabled'] is False and body['max_tokens']==384
            row={**item,'status':'attempted'};result['calls'].append(row);write('EXECUTION.json',result)
            t=time.monotonic()
            req=urllib.request.Request('https://openrouter.ai/api/v1/chat/completions',raw,
                {'Content-Type':'application/json','Authorization':'Bearer '+key})
            with urllib.request.urlopen(req,timeout=30) as response: answer=response.read()
            (OUT/(name+'.response.json')).write_bytes(answer)
            r=json.loads(answer);choice=r['choices'][0];usage=r['usage'];msg=choice['message']
            row.update(seconds=time.monotonic()-t,usage=usage,provider=r.get('provider'),finish_reason=choice.get('finish_reason'))
            assert r.get('provider')=='CoreWeave' and choice['finish_reason']=='stop'
            assert not(msg.get('reasoning') or msg.get('reasoning_content'))
            assert usage.get('completion_tokens_details',{}).get('reasoning_tokens',0)==0
            assert isinstance(usage.get('cost'),(int,float))
            result['cost_usd']+=usage['cost'];assert result['cost_usd']<.25
            try:
                critique=validate_critique(decode_json(msg['content']))
                source=json.dumps(body['messages'][1]['content'])
                # Decode escaped input before exact source binding.
                material=json.loads(body['messages'][1]['content'])
                def strings(x):
                    if isinstance(x,str):return [x]
                    if isinstance(x,dict):return sum((strings(v) for v in x.values()),[])
                    if isinstance(x,list):return sum((strings(v) for v in x),[])
                    return []
                quote=critique['actor_quote']
                if quote: assert any(quote in s for s in strings(material)), 'quote not found'
                if item['arm']=='new' and critique['assessment']=='no_concern':
                    assert all(not v for k,v in critique.items() if k!='assessment')
                row.update(status='valid',critique=critique)
            except Exception as e:row.update(status='output_failure',error=str(e))
            write('EXECUTION.json',result)
            print(name,row['status'],flush=True)
        result['status']='completed'
    except Exception as e:result.update(status='aborted',error=f'{type(e).__name__}: {e}')
    write('EXECUTION.json',result)
    print(json.dumps({'status':result['status'],'cost':result['cost_usd']}))

if __name__=='__main__':
    {'prepare':prepare,'execute':execute}[sys.argv[1]]()
