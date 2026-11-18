"""Bounded real solver/game contact. Invoke only after local tests, with --execute."""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / 'analysis/deployment-probes/20260907-designer-player-openrouter'
MODEL = 'qwen/qwen3.8-27b'


def append(name, row):
    with (OUT/name).open('a') as f:
        f.write(json.dumps(row, allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    from freeze_candidate import verify
    manifest = verify()
    if not args.execute:
        print(json.dumps({'source_verified':manifest['contract_sha256'], 'paid_calls':0})); return
    if (OUT/'STARTED.json').exists():
        raise RuntimeError('Never reuse a started run; a retry needs a new card and directory')
    for p in (ROOT/'.env', Path('/home/pranav/projects/arc-agi-3/.env')):
        if p.exists():
            for line in p.read_text().splitlines():
                if '=' in line and not line.lstrip().startswith('#'):
                    k,v=line.split('=',1); os.environ.setdefault(k.strip(),v.strip().strip('\"\''))
    key = os.environ['OPENROUTER_API_KEY']
    for name in list(os.environ):
        if 'KEY' in name or 'TOKEN' in name:
            del os.environ[name]
    os.environ.update(LITELLM_LOCAL_MODEL_COST_MAP='True', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
        MPLCONFIGDIR=str(OUT/'mpl'), ARC_DISABLE_DOTENV='1', ARC_LLM_MODEL=MODEL, ARC_LLM_API_KEY='local-relay',
        ARC_LLM_EXTRA_BODY='{"reasoning":{"enabled":false}}', ARC_STALL_REVIEW='0',
        TYCHO_KAGGLE_MAX_ACTIONS='10', TYCHO_NOOA_WORKER_TIMEOUT_SECONDS='150',
        NOOA_AGENT_READY_TIMEOUT_SECONDS='30', NOOA_JOIN_ROOT=str(HERE),
        NOOA_AGENT_COMMAND=json.dumps([sys.executable,str(HERE/'headless.py'),'--run-dir','.']))
    state = dict(reserved=0., calls=0, arm='', stop=None, fatal=False)
    lock = threading.Lock()
    def abort(reason):
        state['stop']=reason
        for ipc in (OUT/state['arm']).rglob('ipc'):
            (ipc/'ABORT').write_text(reason)
    class Relay(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_POST(self):
            length=int(self.headers.get('Content-Length','0'))
            if length > 120000:
                abort('input cap'); self.send_error(429); return
            data=self.rfile.read(length)
            with lock:
                reserve=length*.00000042 + 4096*.000003 + .01
                if state['calls']>=12 or state['reserved']+reserve>2 or state['stop']:
                    abort('call/spend cap'); self.send_error(429); return
                payload=json.loads(data)
                payload.update(model=MODEL,max_tokens=4096,stream=False,reasoning={'enabled':False},
                               temperature=1.0,top_p=.95,top_k=20,
                               provider={'max_price':{'prompt':.42,'completion':3,'request':0},'allow_fallbacks':False,'sort':'price','require_parameters':True})
                state['calls']+=1; state['reserved']+=reserve
                identity=dict(arm=state['arm'],call=state['calls'],time=time.time())
                append('requests.jsonl',{**identity,'request':payload,'reservation':reserve})
                try:
                    request=urllib.request.Request('https://openrouter.ai/api/v1/chat/completions',
                        data=json.dumps(payload).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
                    with urllib.request.urlopen(request,timeout=120) as response: raw=response.read()
                    parsed=json.loads(raw)
                    append('responses.jsonl',{**identity,'response':parsed})
                    if parsed.get('error') or not parsed.get('choices'):
                        raise ValueError('provider did not return choices')
                    reasoning=parsed.get('usage',{}).get('completion_tokens_details',{}).get('reasoning_tokens',0)
                    if reasoning:
                        state['fatal']=True; raise ValueError('unexpected reasoning tokens')
                    self.send_response(200); self.send_header('Content-Type','application/json'); self.end_headers(); self.wfile.write(raw)
                except Exception as error:
                    append('errors.jsonl',{**identity,'type':type(error).__name__,'message':str(error)})
                    abort('upstream failure; no forwarding retry'); self.send_error(502)
    server=HTTPServer(('127.0.0.1',0),Relay)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    os.environ['ARC_LLM_BASE_URL']=f'http://127.0.0.1:{server.server_port}/v1'
    sys.path[:0]=[str(ROOT/'deployment/kaggle/tests'),str(ROOT/'deployment/kaggle')]
    # Match the packaged adapter and shared transport, not mutable loose-source defaults.
    spec=importlib.util.spec_from_file_location('runtime.local_solver',HERE/'shared_local_solver.py')
    runtime=importlib.util.module_from_spec(spec)
    sys.modules['runtime.local_solver']=runtime; spec.loader.exec_module(runtime)
    import pytest
    import test_my_agent
    test_my_agent.AGENT_PATH=HERE/'adapter.py'
    import test_nooa_join as join
    import test_join_qualification as qual
    from arc_agi import Arcade, OperationMode
    from arcengine import FrameData, FrameDataRaw
    join.CANDIDATE=HERE
    source=ROOT/'analysis/deployment-probes/20260904T-nooa-join-public3-depth/kaggle-output/ARC-AGI-3-Agents/agents/agent.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest()=='85655f7a11a105e50bba76efd00bebf8056d02ddbae827a8f8a5b46e849fc087'
    cls=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='Agent')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_convert_raw_frame_data')
    ns={'FrameData':FrameData,'FrameDataRaw':FrameDataRaw}
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(source),'exec'),ns)
    def convert(raw): return ns['_convert_raw_frame_data'](None,raw)
    (OUT/'STARTED.json').write_text(json.dumps({'manifest':manifest,'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'card_sha256':hashlib.sha256((OUT/'EXPERIMENT-CARD.md').read_bytes()).hexdigest(),'started':time.time()},indent=2))
    results=[]
    try:
        for arm in ('loop','mindset'):
            verify()
            path=OUT/arm; path.mkdir(exist_ok=False)
            state.update(arm=arm,calls=0,stop=None)
            os.environ['ARC_SOLVER_ARM']=arm
            arcade=Arcade(operation_mode=OperationMode.OFFLINE,environments_dir=str(ROOT/'environment_files'),recordings_dir=str(path/'recordings'))
            env=arcade.make('ls20'); current=convert(env.reset())
            patch=pytest.MonkeyPatch(); agent=qual.build_agent(patch,path,current)
            start=time.monotonic(); error=None
            def timeout(*_): raise TimeoutError('600s arm cap')
            signal.signal(signal.SIGALRM,timeout); signal.alarm(600)
            try:
                for _ in range(22):
                    if agent.is_done(agent.frames,current): break
                    action=agent.choose_action(agent.frames,current); agent.action_counter+=1
                    current=convert(env.step(action)); agent.frames.append(current)
            except Exception as exc: error=f'{type(exc).__name__}: {exc}'
            finally:
                signal.alarm(0)
                try: agent.cleanup(scorecard=None)
                except Exception as exc: error=error or f'cleanup: {exc}'
                patch.undo()
            result=dict(arm=arm,decisions=agent._solver_decisions_emitted,levels=current.levels_completed,
                        calls=state['calls'],wall_seconds=time.monotonic()-start,error=error,stop=state['stop'],reserved_total=state['reserved'])
            results.append(result); (OUT/'RESULTS.json').write_text(json.dumps(results,indent=2)+'\n')
            print(json.dumps(result),flush=True)
            if state['fatal']: break
    finally:
        server.shutdown()


if __name__=='__main__': main()
