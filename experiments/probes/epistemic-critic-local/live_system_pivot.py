"""Matched restarted Actor runs: system-level correction speed and retention."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

from live_pilot import ROOT, RUNTIME, MODEL, dump
from paired_commitment import RESPONSE_TEMPLATE, validate_critique, quote_binding
CRITIC_PROMPT = (Path(__file__).parent / "framework-v1/system.md").read_text()
CRITIC_PROMPT += "\nIdentify the most consequential commitment that should change now. If the evidence warrants a change, state the smallest correction. Otherwise let the Actor continue."
sys.path.insert(0, str(Path(__file__).parent / "system-pivot-v1"))
import fixture as pivot_fixture
FIXTURE = None
from claim_screen import decode_json


class AdmissionStop(RuntimeError): pass


class Budget:
    def __init__(self,limit=.5): self.limit=limit;self.spent=0.;self.pending=0.
    def reserve(self,value):
        if self.spent+self.pending+value>self.limit:raise AdmissionStop('dollar cap')
        self.pending+=value
    def settle(self,reservation,cost):
        if not isinstance(cost,(float,int)) or not 0<=cost<=reservation:
            raise ValueError('unresolved or out-of-reservation cost')
        self.pending-=reservation;self.spent+=cost


def review_due(actor_calls,actions,last_calls,last_actions,reviews):
    return reviews<20 and actor_calls>0 and (actor_calls-last_calls>=2 or actions-last_actions>=2)


def environment(path):
    return pivot_fixture.environment(path, FIXTURE)


def snapshot(path,label):
    """Copy helper/notes evidence at paused HTTP boundary; do not execute helpers."""
    result={}
    for workspace in (path/'workspaces').glob('*'):
        for folder in ('helpers','knowledge'):
            source=workspace/'team_nemo/shared'/folder
            for p in source.rglob('*') if source.exists() else []:
                if not p.is_file() or '__pycache__' in p.parts:continue
                target=path/'snapshots'/label/folder/p.relative_to(source)
                target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
                result[str(target.relative_to(path))]=hashlib.sha256(target.read_bytes()).hexdigest()
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--execute',action='store_true');p.add_argument('--fixture',type=Path,required=True);p.add_argument('--order',choices=['baseline-first','critic-first'],required=True);args=p.parse_args();out=args.output.resolve()
    global FIXTURE
    FIXTURE=args.fixture.resolve()
    config=json.loads((out/'CONFIG.json').read_text())
    if config['order']!=args.order or config['fixture_sha256']!=hashlib.sha256((FIXTURE/'evidence.json').read_bytes()).hexdigest():
        raise RuntimeError('Prepared order or fixture differs from requested run')
    for name,expected in config['source_sha256'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=expected:
            raise RuntimeError('Prepared source drift: '+name)
    out.mkdir(parents=True,exist_ok=True)
    if (out/'STARTED.json').exists():raise RuntimeError('No relaunch of a started run')
    spec=importlib.util.spec_from_file_location('endurance_freeze',RUNTIME/'freeze_candidate.py')
    frozen=importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen);manifest=frozen.verify()
    _,first,_=environment(out/'preflight-a');_,second,_=environment(out/'preflight-b')
    assert first.frame==second.frame and first.levels_completed==second.levels_completed
    dump(out/'PREFLIGHT.json',{'initial_grids_match':True,'paid_calls':0,'runtime':manifest['contract_sha256'],'fixture_sha256':hashlib.sha256((FIXTURE/'evidence.json').read_bytes()).hexdigest(),'order':args.order})
    if not args.execute:print('Fresh initial grids match; frozen runtime verifies; no inference.');return
    assert (out/'EXPERIMENT-CARD.md').exists()
    key=os.environ.get('OPENROUTER_API_KEY')
    for file in (ROOT/'.env',Path('/home/pranav/projects/arc-agi-3/.env')):
        if key:break
        if file.exists():
            for line in file.read_text().splitlines():
                if line.startswith('OPENROUTER_API_KEY='):key=line.split('=',1)[1].strip().strip('\"\'')
    if not key:raise RuntimeError('No existing credential')
    for name in list(os.environ):
        if 'KEY' in name or 'TOKEN' in name:del os.environ[name]
    os.environ.update(LITELLM_LOCAL_MODEL_COST_MAP='True',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
        MPLCONFIGDIR=str(out/'mpl'),ARC_DISABLE_DOTENV='1',ARC_LLM_MODEL=MODEL,ARC_LLM_API_KEY='local-relay',
        ARC_SOLVER_ARM='loop',ARC_DISCOVERY_ARM='a',ARC_STALL_REVIEW='0',ARC_CONTINGENT_ARM='off',
        ARC_PROCEDURAL_DIAGNOSTIC='0',ARC_LLM_MAX_TOKENS='4096',ARC_LLM_TIMEOUT_SECONDS='125',
        ARC_LLM_EXTRA_BODY='{"reasoning":{"enabled":false},"chat_template_kwargs":{"enable_thinking":false}}',
        TYCHO_KAGGLE_MAX_ACTIONS='64',TYCHO_NOOA_WORKER_TIMEOUT_SECONDS='610',
        NOOA_AGENT_READY_TIMEOUT_SECONDS='30',NOOA_JOIN_ROOT=str(RUNTIME),
        NOOA_AGENT_COMMAND=json.dumps([sys.executable,str(RUNTIME/'headless.py'),'--run-dir','.']))
    state={'arm':'','closed':True,'calls':0,'actor_calls':0,'reviews':0,'actions':0,
           'last_review_calls':0,'last_review_actions':0,'last_actor':None,'fatal':None,'cap':None}
    idle=threading.Event();idle.set()
    def append(file,row):
        with (out/file).open('a') as f:f.write(json.dumps(row)+'\n')
    def forward(body,purpose):
        if state['closed'] or time.monotonic()>=state['deadline']:raise AdmissionStop('time/closed admission')
        if purpose=='actor' and state['actor_calls']>=96:raise AdmissionStop('Actor call cap')
        body=copy.deepcopy(body);body.pop('parallel_tool_calls',None)
        body.update(model=MODEL,stream=False,temperature=1.,top_p=.95,top_k=20,
            provider={'only':['coreweave/fp8'],'allow_fallbacks':False,'require_parameters':True,
                      'max_price':{'prompt':.42,'completion':3,'request':0}})
        assert body.get('reasoning',{}).get('enabled') is False
        raw=json.dumps(body).encode()
        if len(raw)>1000000:raise AdmissionStop('input byte cap')
        reserve=(len(raw)+4096)*.42/1e6+body['max_tokens']*3/1e6
        budget=state['budget'];budget.reserve(reserve)
        state['calls']+=1;state['actor_calls']+=purpose=='actor'
        identity={'call':state['calls'],'arm':state['arm'],'purpose':purpose,'actions_before':state['actions'],'arm_elapsed_seconds':time.monotonic()-state['arm_started']}
        append('requests.jsonl',{**identity,'request':body,'reservation':reserve})
        start=time.monotonic()
        req=urllib.request.Request('https://openrouter.ai/api/v1/chat/completions',raw,
            {'Content-Type':'application/json','Authorization':'Bearer '+key})
        try:
            with urllib.request.urlopen(req,timeout=30 if purpose=='critic' else 90) as response:reply=response.read()
            result=json.loads(reply)
        except Exception as exc:
            append('transport-errors.jsonl',{**identity,'error':str(exc),
                'body':exc.read().decode(errors='replace') if hasattr(exc,'read') else None})
            raise
        elapsed=time.monotonic()-start
        append('responses.jsonl',{**identity,'response':result,'elapsed_seconds':elapsed,'arm_elapsed_seconds':time.monotonic()-state['arm_started']})
        usage=result.get('usage',{});budget.settle(reserve,usage.get('cost'))
        if result.get('provider')!='CoreWeave':raise RuntimeError('provider mismatch')
        msg=result['choices'][0]['message']
        if msg.get('reasoning') or msg.get('reasoning_content') or usage.get('completion_tokens_details',{}).get('reasoning_tokens',0):
            raise RuntimeError('unexpected reasoning generation')
        if usage.get('completion_tokens',0)>body['max_tokens']:raise RuntimeError('token budget exceeded')
        return result,identity
    class Relay(BaseHTTPRequestHandler):
        def log_message(self,*_):pass
        def do_POST(self):
            idle.clear()
            try:
                size=int(self.headers.get('Content-Length','0'))
                if size>1000000:raise AdmissionStop('input byte cap')
                body=json.loads(self.rfile.read(size))
                if body.get('max_tokens')!=4096:raise RuntimeError('Actor budget mismatch')
                path=out/state['arm'];label=f'before-actor-{state["actor_calls"]+1:03d}'
                snap=snapshot(path,label)
                append('snapshots.jsonl',{'arm':state['arm'],'label':label,'files':snap})
                if state['arm']=='critic' and review_due(state['actor_calls'],state['actions'],state['last_review_calls'],state['last_review_actions'],state['reviews']):
                    state['reviews']+=1;state['last_review_calls']=state['actor_calls'];state['last_review_actions']=state['actions']
                    rid=f'critic:{state["reviews"]}'
                    material=json.dumps(state['last_actor'],ensure_ascii=False)
                    cb=copy.deepcopy(body);cb.pop('tools',None);cb.pop('tool_choice',None);cb['max_tokens']=384
                    cb['messages']=[{'role':'system','content':CRITIC_PROMPT},
                        {'role':'user','content':json.dumps({'actor_request_as_evidence':body,
                            'review_material':material,'scope':'Review the latest Actor reasoning in its full context; no fault is stipulated.'})}]
                    append('review-contexts.jsonl',{'review_id':rid,'actor_request':body,'source_actor':state['last_actor'],
                        'context_sha256':hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest(),'snapshot':label})
                    result,identity=forward(cb,'critic')
                    try:
                        choice=result['choices'][0];msg=choice['message']
                        if choice['finish_reason']!='stop' or msg.get('tool_calls'):raise ValueError('unfinished critique or tool attempt')
                        critique=validate_critique(decode_json(msg.get('content') or ''))
                        # Search decoded Actor code/content so JSON escaping does not corrupt source binding.
                        sources=[];last=state['last_actor'] or {}
                        if last.get('content'):sources.append(('content',last['content']))
                        for i,t in enumerate(last.get('tool_calls') or []):
                            try:sources.append((f'tool_calls[{i}].code',json.loads(t['function']['arguments'])['code']))
                            except (ValueError,KeyError):pass
                        matches=[{'field':field,**quote_binding(critique['actor_quote'],text)} for field,text in sources]
                        count=sum(len(m['spans']) for m in matches)
                        if critique['actor_quote'] and count!=1:raise ValueError('quote not uniquely located in latest Actor output')
                        append('reviews.jsonl',{**identity,'review_id':rid,'status':'completed','critique':critique,'source_bindings':matches})
                        body['messages'].append({'role':'user','content':RESPONSE_TEMPLATE.format(review_id=rid,critique=json.dumps(critique))})
                    except ValueError as exc:
                        append('reviews.jsonl',{**identity,'review_id':rid,'status':'invalid_output','error':str(exc)})
                    # Advice enters one response only; no repeated stale endorsement.
                response,identity=forward(body,'actor')
                if response['choices'][0]['finish_reason'] not in ('stop','tool_calls'):
                    raise RuntimeError('Actor completion truncated')
                state['last_actor']=response['choices'][0]['message']
                self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers()
                self.wfile.write(json.dumps(response).encode())
            except AdmissionStop as exc:
                state['closed']=True;state['cap']=str(exc)
                append('stops.jsonl',{'arm':state['arm'],'reason':str(exc)})
                try:self.send_error(429,str(exc))
                except (BrokenPipeError,ConnectionResetError):pass
            except (BrokenPipeError,ConnectionResetError):
                append('delivery.jsonl',{'arm':state['arm'],'closed':state['closed'],'status':'client retired'})
            except Exception as exc:
                state['closed']=True;state['fatal']=str(exc)
                append('errors.jsonl',{'arm':state['arm'],'error':type(exc).__name__,'detail':str(exc)})
                try:self.send_error(502)
                except (BrokenPipeError,ConnectionResetError):pass
            finally:idle.set()
    server=HTTPServer(('127.0.0.1',0),Relay)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    os.environ['ARC_LLM_BASE_URL']=f'http://127.0.0.1:{server.server_port}/v1'
    sys.path[:0]=[str(ROOT/'deployment/kaggle/tests'),str(ROOT/'deployment/kaggle')]
    spec=importlib.util.spec_from_file_location('runtime.local_solver',RUNTIME/'shared_local_solver.py')
    runtime=importlib.util.module_from_spec(spec);sys.modules['runtime.local_solver']=runtime;spec.loader.exec_module(runtime)
    import pytest
    import test_my_agent
    test_my_agent.AGENT_PATH=RUNTIME/'adapter.py'
    import test_nooa_join as join
    join.CANDIDATE=RUNTIME
    import test_join_qualification as qual
    own=Path(__file__).resolve().parent
    with (out/'STARTED.json').open('x') as f:
        json.dump({'time':time.time(),'runtime':manifest,'files':{str(p):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__).resolve(),own/'paired_commitment.py',own/'reasoning_screen.py',own/'claim_screen.py',own/'framework-v1/system.md',own/'live_hook.py',own/'system-pivot-v1/fixture.py',own/'system-pivot-v1/score.py',FIXTURE/'seed.txt',FIXTURE/'evidence.json',FIXTURE/'MANIFEST.json',out/'EXPERIMENT-CARD.md',out/'CONFIG.json']}},f,indent=2)
    results=[]
    try:
        for arm in (('baseline','critic') if args.order=='baseline-first' else ('critic','baseline')):
            frozen.verify();path=out/arm;path.mkdir()
            env,current,convert=environment(path)
            assert current.frame==first.frame
            dump(path/'initial.json',current.model_dump(mode='json'))
            state.update(arm=arm,closed=False,actor_calls=0,reviews=0,actions=0,last_review_calls=0,
                last_review_actions=0,last_actor=None,fatal=None,cap=None,budget=Budget(),deadline=time.monotonic()+600)
            os.environ['NOOA_AGENT_COMMAND']=json.dumps([sys.executable,str(own/'live_hook.py'),str(RUNTIME),'baseline',str(FIXTURE/'seed.txt'),'unused'])
            start=time.monotonic();state['arm_started']=start;state['deadline']=start+600
            patch=pytest.MonkeyPatch();agent=qual.build_agent(patch,path,current)
            error=None
            def timeout(*_):
                state.update(closed=True,cap='600s arm deadline')
                raise TimeoutError('600s arm deadline')
            signal.signal(signal.SIGALRM,timeout);signal.alarm(600)
            try:
                for step in range(64):
                    if agent.is_done(agent.frames,current):break
                    action=agent.choose_action(agent.frames,current);agent.action_counter+=1
                    before=current.frame[-1];current=convert(env.step(action, data=action.action_data.model_dump(mode="json")));agent.frames.append(current)
                    state['actions']+=1
                    append('actions.jsonl',{'arm':arm,'step':step+1,'action':action.name,'action_data':action.action_data.model_dump(mode='json'),'before':before,
                        'after':current.frame[-1],'levels_completed':current.levels_completed,
                        'state':current.state.name,'elapsed_seconds':time.monotonic()-start})
                    print(json.dumps({'arm':arm,'step':step+1,'action':action.name,'levels':current.levels_completed,
                        'actor_calls':state['actor_calls'],'reviews':state['reviews'],'cost':state['budget'].spent}),flush=True)
            except Exception as exc:error=f'{type(exc).__name__}: {exc}'
            finally:
                signal.alarm(0);state['closed']=True
                try:agent.cleanup(scorecard=None)
                except Exception as exc:error=error or f'cleanup: {exc}'
                patch.undo()
                if not idle.wait(95):state['fatal']='HTTP transport retirement unresolved'
                snapshot(path,'final')
            result={'arm':arm,'actions':state['actions'],'levels_completed':current.levels_completed,'game':'ls20',
                'actor_calls':state['actor_calls'],'reviews':state['reviews'],'seconds':time.monotonic()-start,
                'actual_cost':state['budget'].spent,'unsettled_reservation':state['budget'].pending,
                'cap':state['cap'],'fatal':state['fatal'],'error':error}
            results.append(result);dump(out/'RESULTS.json',results);print(json.dumps(result),flush=True)
            if state['fatal']:break
    finally:
        server.shutdown();server.server_close();thread.join(2)


if __name__=='__main__':main()
