"""CPU-only real packaged LocalSolver/worker/NOOA against loopback scripted completion."""
import concurrent.futures, hashlib, importlib.util, json, os, sys, threading, time
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
PKG=ROOT/'deployment/kaggle/q6-summary-boundary-20260906'
DEPS=str(ROOT/'solvers/tycho_nooa/src')+':/home/pranav/projects/arc-agi-3-reference/tycho'
sys.path[:0]=DEPS.split(':')
spec=importlib.util.spec_from_file_location('pinned_local_solver',PKG/'shared_local_solver.py')
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
from tycho_nooa.models import GameTurn
from tycho_nooa.protocol import ObservationPayload,ActionResultMessage,project_arc_observation
lock=threading.Lock(); events=[]; active=0; peak=0; counts={}; second_entered=threading.Event(); release=threading.Event()
def log(**kw):
    with lock:
        events.append(dict(at=time.monotonic(),**kw))
        (HERE/'events.json').write_text(json.dumps(events,indent=2))
class Handler(BaseHTTPRequestHandler):
    def log_message(self,*a): pass
    def do_POST(self):
        global active,peak
        body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        owner=body['probe_owner']
        with lock:
            counts[owner]=counts.get(owner,0)+1;n=counts[owner];active+=1;peak=max(peak,active)
        (HERE/f'request-{owner}-{n}.json').write_text(json.dumps(body,indent=2))
        log(event='http_begin',owner=owner,n=n,path=self.path)
        if owner=='actor_a' and n==2:
            second_entered.set();release.wait(30)
        if owner=='actor_b': release.set()
        code="self.submit_actions(['UP'], rationale='predict: grid changes', predicted_path=[{'expect':'move','z':None}])"
        result={'id':f'local-{owner}-{n}','object':'chat.completion','created':int(time.time()),'model':owner,'choices':[{'index':0,'message':{'role':'assistant','content':None,'tool_calls':[{'id':f'call_{n}','type':'function','function':{'name':'execute_python','arguments':json.dumps({'code':code})}}]},'finish_reason':'tool_calls'}],'usage':{'prompt_tokens':100,'completion_tokens':30,'total_tokens':130}}
        data=json.dumps(result).encode();self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        log(event='http_end',owner=owner,n=n)
        with lock: active-=1
server=ThreadingHTTPServer(('127.0.0.1',0),Handler);threading.Thread(target=server.serve_forever,daemon=True).start()
def obs(v):return project_arc_observation(frames=[[v]],state='NOT_FINISHED',levels_completed=0,available_action_ids=[1],full_reset=False)
def turn(v):return GameTurn(grid=[[v]],available_actions=['RESET','UP'],sequence=0,level=0,turn_in_level=0,levels_completed=0,state='NOT_FINISHED')
solvers=[]
try:
    for owner,v in [('actor_a',1),('actor_b',3)]:
        env={'PATH':os.environ['PATH'],'PYTHONPATH':DEPS,'PYTHONUNBUFFERED':'1','LITELLM_LOCAL_MODEL_COST_MAP':'True','ARC_LLM_MODEL':'shared_actor_probe','ARC_LLM_EXTRA_BODY':json.dumps({'probe_owner':owner}),'ARC_LLM_BASE_URL':f'http://127.0.0.1:{server.server_port}/v1','ARC_LLM_API_KEY':'local-script-only','ARC_LLM_USE_RESPONSES':'0'}
        solver=m.LocalSolver(game_id=owner,workspace_root=HERE/'sessions',initial_observation=obs(v),command=[sys.executable,str(PKG/'nooa_worker.py')],timeout_seconds=45,environment=env)
        solvers.append(solver)
    a,b=solvers
    decision=a.decide(turn(1));log(event='a_first_decision')
    a.observe(ActionResultMessage(protocol_version=3,type='action_result',sequence=0,level=0,turn_in_level=0,decision=decision,before=obs(1),after=obs(2)))
    log(event='a_observe_returned')
    assert second_entered.wait(15),'A did not autonomously request successor'
    # No second A decide is sent. Its successor request must originate from observe publication.
    decision_b=b.decide(turn(3));log(event='b_first_decision',decision=decision_b.model_dump())
    assert peak>=2,peak
    result={'status':'OVERLAPPING_REAL_NOOA_HTTP_REQUESTS','peak_http_inflight':peak,'a_second_decide_called':False,'endpoint':f'http://127.0.0.1:{server.server_port}/v1/chat/completions','counts_at_assertion':dict(counts),'model':'shared_actor_probe','sessions':[str(s.session_root) for s in solvers],'source_sha256':hashlib.sha256((PKG/'shared_local_solver.py').read_bytes()).hexdigest(),'limits':'Scripted HTTP outputs, synthetic public grids, no GPU or actual game; does not establish backend batching/performance.'}
    (HERE/'RESULT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
finally:
    release.set()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda s:s.close(),solvers))
    server.shutdown()
