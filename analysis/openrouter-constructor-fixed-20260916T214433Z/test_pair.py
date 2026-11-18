"""Loopback provider fixtures through real constructor and sealed execution."""
import copy, json, sys, threading, time
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
import pytest
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE/'engine')]
from run_pair import episode, supervise
from transport import initial_ledger, write, Transport, Stop, wire, MODEL, PROVIDER
from constructor import construction_messages, ConstructionLimits, TOOLS
from prefix import project_transitions
from constructor_contract import runtime_contract
SOURCE='''def initialize(observation):
 return {"grid":observation["grid"]}
def predict(state,action):
 return [{"state":None,"commitments":{}}]
def assimilate(predicted_state,observation,verdict):
 return {"grid":observation["grid"]}
'''
WORK={'decision':'fixture next action','obstacle':'fixture unknown','operation':'test','claim':'ABI fixture','evidence_refs':[1],'dependencies':['grid from observation'],'completion_checks':[1],'remaining_scope':'all dynamics','reasoning_need':'needs_computation'}
DETAILS=[{'type':'reasoning.text','text':'fixture internal deliberation','signature':'opaque-signature','id':'r0','format':'unknown','index':0}]

def response(name,args,ordinal=1,arm='off',reasoning='details'):
    msg={'role':'assistant','content':None,'refusal':None,'tool_calls':[{'index':0,'id':f'opaque-{ordinal}','type':'function','function':{'name':name,'arguments':json.dumps(args)}}]}
    if arm=='on':
        if reasoning=='details':msg['reasoning_details']=copy.deepcopy(DETAILS)
        else:msg['reasoning']='fixture reasoning text'
    return {'id':f'gen-fixture-{ordinal}','provider':'Reka','model':MODEL,'choices':[{'index':0,'finish_reason':'tool_calls','message':msg}],'usage':{'prompt_tokens':100,'completion_tokens':50,'total_tokens':150,'completion_tokens_details':{'reasoning_tokens':10 if arm=='on' else 0},'cost':0.0001489},'openrouter_metadata':{'requested':MODEL,'strategy':'direct','attempt':1,'pipeline':[],'endpoints':{'available':[{'provider':'Reka','model':MODEL,'selected':True}]}}}

class Server:
    def __init__(self,replies):self.replies=list(replies);self.requests=[]
    def __enter__(self):
        outer=self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                outer.requests.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
                reply=outer.replies.pop(0)
                if isinstance(reply,tuple):status,reply=reply
                else:status=200
                raw=reply if isinstance(reply,bytes) else json.dumps(reply).encode()
                self.send_response(status);self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
        self.server=HTTPServer(('127.0.0.1',0),Handler);self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start();self.url=f'http://127.0.0.1:{self.server.server_port}/chat';return self
    def __exit__(self,*args):self.server.shutdown();self.server.server_close();self.thread.join()

def body():
    return {'model':MODEL,'messages':construction_messages(project_transitions(HERE/'prefix.jsonl'),None),'tools':TOOLS,'tool_choice':'required','provider':{},'max_tokens':4096,'reasoning':{'enabled':False}}

@pytest.mark.parametrize('arm,representation',[('off','details'),('on','details'),('on','string')])
def test_real_constructor_inspection_failed_submission_repair_freeze(tmp_path,arm,representation):
    ledger=tmp_path/'ledger.json';initial_ledger(ledger)
    responses=[response('execute_python',{'code':'import json\nc=json.load(open("/evidence/case.json"))\nprint(c["frontier"])'},1,arm,representation),response('submit_candidate',{'work_item':WORK,'candidate_source':repr(SOURCE.splitlines())},2,arm,representation),response('submit_candidate',{'work_item':WORK,'candidate_source':SOURCE.splitlines()},3,arm,representation)]
    with Server(responses) as server:
        r=episode(arm,tmp_path/'arm',ledger,fixture_url=server.url)
    assert r['frozen'] and r['submissions']==2 and r['inspections']==1
    assert r['repair']=={'failed_submissions':1,'accepted_after_repair':True}
    requests=server.requests;assert len(requests)==3
    for req in requests:
        assert req['provider']==PROVIDER and req['max_tokens']==4096
        assert req['reasoning']==({'enabled':True,'exclude':False,'effort':'medium'} if arm=='on' else {'enabled':False,'exclude':False})
        assert 'parallel_tool_calls' not in req
    assistant=next(m for m in requests[1]['messages'] if m['role']=='assistant')
    assert assistant['content'] is None and 'index' not in assistant['tool_calls'][0]
    assert 'refusal' not in assistant
    if arm=='on':
        assert (assistant.get('reasoning_details')==DETAILS if representation=='details' else assistant.get('reasoning')=='fixture reasoning text')
    tool=[m for m in requests[2]['messages'] if m['role']=='tool'][-1]
    assert 'NameError' in tool['content'] and 'initialize' in tool['content']
    assert json.loads((tmp_path/'arm/FROZEN.json').read_text())['candidate_source']==SOURCE
    l=json.loads(ledger.read_text());assert l['unknown_exposure_usd']=='0' and not l['stopped']
    assert l['arms'][arm]['requests']==3 and len(l['attempts'])==3
    assert all(float(x['exposure_usd'])>0.334 for x in l['attempts'])

@pytest.mark.parametrize('mode',['first_http','malformed_json','missing_usage','invalid_cost','missing_reasoning','off_reasoning','wrong_provider','fallback','pipeline','overrun','truncated','wrong_total','null_choice'])
def test_real_constructor_failure_accounting_no_retry(tmp_path,mode):
    ledger=tmp_path/'ledger.json';initial_ledger(ledger)
    r=response('execute_python',{'code':'print(1)'})
    if mode=='first_http':r=(429,{'error':{'code':429,'message':'fixture rate limit'}})
    if mode=='malformed_json':r=b'{'
    if mode=='missing_usage':r.pop('usage')
    if mode=='invalid_cost':r['usage']['cost']=-1
    if mode=='missing_reasoning':r['usage'].pop('completion_tokens_details')
    if mode=='off_reasoning':r['choices'][0]['message']['reasoning_details']=DETAILS
    if mode=='wrong_provider':r['provider']='Chutes'
    if mode=='fallback':r['openrouter_metadata']['attempt']=2
    if mode=='pipeline':r['openrouter_metadata']['pipeline']=[{'type':'context_compression'}]
    if mode=='overrun':r['usage'].update(completion_tokens=4097,total_tokens=4197)
    if mode=='truncated':r['choices'][0]['finish_reason']='length'
    if mode=='wrong_total':r['usage']['total_tokens']=999
    if mode=='null_choice':r['choices']=None
    with Server([r]) as server:
        with pytest.raises(Stop):episode('off',tmp_path/'arm',ledger,fixture_url=server.url)
        assert len(server.requests)==1
    l=json.loads(ledger.read_text());assert l['stopped'];assert l['arms']['off']['requests']==1
    unknown=mode in ('first_http','malformed_json','missing_usage','invalid_cost','wrong_total')
    assert (float(l['unknown_exposure_usd'])>0)==unknown
    assert l['attempts'][0]['latency_seconds']>=0
    assert (tmp_path/'arm/raw/001.response.raw').exists()
    assert not (tmp_path/'arm/FROZEN.json').exists()

@pytest.mark.parametrize('mode',['requests','prompt','completion','money','unknown','deadline','context'])
def test_admission_denies_before_http(tmp_path,mode):
    ledger=tmp_path/'ledger.json';initial_ledger(ledger);l=json.loads(ledger.read_text());b=body()
    deadline=time.monotonic()+240
    if mode=='requests':l['arms']['off']['requests']=6
    if mode=='prompt':l['arms']['off']['prompt_tokens']=599999
    if mode=='completion':l['arms']['off']['completion_tokens']=24576
    if mode=='money':l['known_cost_usd']='1.9'
    if mode=='unknown':l['unknown_exposure_usd']='0.1'
    if mode=='deadline':deadline=time.monotonic()+19
    if mode=='context':b['messages'].append({'role':'user','content':'x'*270000})
    write(ledger,l)
    with Server([]) as server:
        t=Transport('off',tmp_path/'raw',ledger,deadline,fixture_url=server.url)
        with pytest.raises(Stop):t(b)
        assert not server.requests
    assert not list((tmp_path/'raw').glob('*.request.json'))

def test_initial_prompt_schema_evidence_parity(tmp_path):
    recorded=[]
    for arm in ('off','on'):
        ledger=tmp_path/f'{arm}.json';initial_ledger(ledger)
        r=response('request_evidence',{'obstacle':'unknown','needed_observation':'another transition','decision_relevance':'fixture'},arm=arm)
        with Server([r]) as server:episode(arm,tmp_path/arm,ledger,fixture_url=server.url)
        recorded.append(server.requests[0])
    a,b=recorded;a.pop('reasoning');b.pop('reasoning');assert a==b
    original=json.loads((HERE/'ORIGINAL-FIRST-MESSAGES.json').read_text())
    assert a['messages'][1]==original[1]
    assert 'Reasoning policy is controlled by the API' in a['messages'][0]['content']
    assert 'source' not in json.loads(a['messages'][1]['content'])['visible_case']

def test_multiple_tool_calls_real_dispatch(tmp_path):
    ledger=tmp_path/'ledger.json';initial_ledger(ledger)
    r=response('execute_python',{'code':'print("first")'})
    r['choices'][0]['message']['tool_calls'].append(response('execute_python',{'code':'print("second")'},2)['choices'][0]['message']['tool_calls'][0])
    final=response('request_evidence',{'obstacle':'x','needed_observation':'y','decision_relevance':'z'},3)
    with Server([r,final]) as server:result=episode('off',tmp_path/'arm',ledger,fixture_url=server.url)
    assert result['inspections']==2
    tools=[m for m in server.requests[1]['messages'] if m['role']=='tool']
    assert [m['tool_call_id'] for m in tools]==['opaque-1','opaque-2']

def test_containment_real_inspection(tmp_path):
    ledger=tmp_path/'ledger.json';initial_ledger(ledger)
    code='''import os, json, socket
results={"uid":os.getuid(),"env":dict(os.environ)}
for name,fn in [("host",lambda:open("/etc/passwd").read()),("network",lambda:socket.socket()),("fork",lambda:os.fork()),("write",lambda:open("/evidence/new","w"))]:
 try:fn();results[name]="ESCAPE"
 except Exception:results[name]="denied"
print(json.dumps(results))'''
    r=response('execute_python',{'code':code});final=response('request_evidence',{'obstacle':'x','needed_observation':'y','decision_relevance':'z'},2)
    with Server([r,final]) as server:episode('off',tmp_path/'arm',ledger,fixture_url=server.url)
    tool=next(m for m in server.requests[1]['messages'] if m['role']=='tool');out=json.loads(json.loads(tool['content'])['stdout'])
    assert out['uid']==65534 and all(out[k]=='denied' for k in ('host','network','fork','write'))
    assert set(out['env'])=={'LANG','PYTHONDONTWRITEBYTECODE'}

def test_hard_kill_after_child_write_no_root_promotion(tmp_path):
    code='from pathlib import Path;import time;p=Path('+repr(str(tmp_path))+');(p/"child-freeze.json").write_text("forensic");time.sleep(30)'
    result=supervise([sys.executable,'-B','-c',code],tmp_path,hard_seconds=.15)
    assert result['hard_timeout'] and result['reaped'] and (tmp_path/'child-freeze.json').exists()
    assert not (tmp_path/'FROZEN.json').exists()

def test_engine_inspection_submission_request_caps_real_wire(tmp_path):
    ledger=tmp_path/'ledger.json';initial_ledger(ledger)
    first=response('execute_python',{'code':'print(1)'})
    first['choices'][0]['message']['tool_calls']=[response('execute_python',{'code':'print(1)'},i)['choices'][0]['message']['tool_calls'][0] for i in range(1,13)]
    # After the inspection cap, commit is irreversible. Four failed submits then stop.
    rest=[response('submit_candidate',{'work_item':WORK,'candidate_source':repr(SOURCE.splitlines())},20+i) for i in range(4)]
    rest.append(response('request_evidence',{'obstacle':'x','needed_observation':'y','decision_relevance':'z'},30))
    with Server([first,*rest]) as server:r=episode('off',tmp_path/'arm',ledger,fixture_url=server.url)
    assert r['inspections']==10 and r['submissions']==4 and r['model_requests']==6
    assert all(t['function']['name']!='execute_python' for req in server.requests[1:] for t in req['tools'])
    assert all(t['function']['name']!='submit_candidate' for t in server.requests[-1]['tools'])
    events=[json.loads(x) for x in (tmp_path/'arm/tool-events.jsonl').read_text().splitlines()]
    assert sum('tool attempt budget exhausted' in x['rendered_result_message']['content'] for x in events)==2

def test_malformed_tool_arguments_existing_feedback(tmp_path):
    ledger=tmp_path/'ledger.json';initial_ledger(ledger)
    first=response('execute_python',{'code':'print(1)'})
    first['choices'][0]['message']['tool_calls'][0]['function']['arguments']='['
    final=response('request_evidence',{'obstacle':'x','needed_observation':'y','decision_relevance':'z'},2)
    with Server([first,final]) as server:r=episode('off',tmp_path/'arm',ledger,fixture_url=server.url)
    assert r['inspections']==1
    tool=next(m for m in server.requests[1]['messages'] if m['role']=='tool')
    assert 'JSONDecodeError' in tool['content']


def test_actual_http_inflight_hard_kill_retains_unknown(tmp_path):
    entered=threading.Event()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']));entered.set();time.sleep(.5)
            try:self.send_response(200);self.end_headers();self.wfile.write(b'{}')
            except (BrokenPipeError,ConnectionResetError):pass
    server=HTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    ledger=tmp_path/'ledger.json';initial_ledger(ledger)
    script='import sys;sys.path.insert(0,'+repr(str(HERE))+');from run_pair import episode;episode("off",'+repr(str(tmp_path/'arm'))+','+repr(str(ledger))+',fixture_url='+repr(f'http://127.0.0.1:{server.server_port}/chat')+')'
    try:receipt=supervise([sys.executable,'-B','-c',script],tmp_path,hard_seconds=.25)
    finally:server.shutdown();server.server_close();thread.join()
    assert entered.is_set() and receipt['hard_timeout']
    d=json.loads(ledger.read_text());assert len(d['attempts'])==1 and float(d['unknown_exposure_usd'])>0
    assert not (tmp_path/'arm/FROZEN.json').exists() and not (tmp_path/'arm/RESULT.json').exists()


def test_inert_credential_loader_does_not_open_file(monkeypatch):
    import auth
    monkeypatch.delenv('OPENROUTER_API_KEY',raising=False)
    def forbidden(*a,**k):raise AssertionError('credential file opened')
    monkeypatch.setattr(Path,'open',forbidden)
    assert auth.load_key()==(None,'credential-use-approval-pending')


def test_gate_denies_missing_or_mismatched_approval(monkeypatch,tmp_path):
    import run_pair
    monkeypatch.setattr(run_pair,'verify_inventory',lambda:'fixture-hash')
    p=tmp_path/'not-approval.json';p.write_text('{}')
    with pytest.raises(Stop):run_pair.gates(p,p)

def test_shared_ledger_pair_start_hash_and_cost(tmp_path):
    ledger=tmp_path/'ledger.json';initial_ledger(ledger)
    for arm in ('off','on'):
        r=response('request_evidence',{'obstacle':'x','needed_observation':'y','decision_relevance':'z'},arm=arm)
        with Server([r]) as server:episode(arm,tmp_path/arm,ledger,fixture_url=server.url)
    l=json.loads(ledger.read_text())
    assert l['arms']['off']['initial_shared_sha256']==l['arms']['on']['initial_shared_sha256']==l['off_initial_shared_sha256']
    assert float(l['known_cost_usd'])==pytest.approx(.0002978)
    assert l['unknown_exposure_usd']=='0' and [r['arm'] for r in l['attempts']]==['off','on']


def test_parity_drift_stops_before_second_arm_send(tmp_path):
    ledger=tmp_path/'ledger.json';initial_ledger(ledger)
    l=json.loads(ledger.read_text());l['off_initial_shared_sha256']='different';write(ledger,l)
    with Server([]) as server:
        t=Transport('on',tmp_path/'raw',ledger,time.monotonic()+240,fixture_url=server.url)
        t.fixture=False # Exercise the production parity gate; network still loopback.
        with pytest.raises(Stop,match='parity'):t(body())
        assert not server.requests
    assert json.loads(ledger.read_text())['arms']['on']['requests']==0
