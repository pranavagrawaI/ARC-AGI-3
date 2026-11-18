"""Local only: real provider-shaped HTTP, official game, and sealed generated code."""
import copy,json,sys,time
from pathlib import Path
import pytest
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE/'engine')]
from common import read,put,hash_file,encoded,digest,CAPS
from fixtures import Server,response,construction,SOURCE,WORK,DETAILS
from run_utility import run_pair,supervise,promote
from transport import Transport,Stop,initial_ledger,write,wire,PROVIDER
from actor import ACT,SYSTEM

def body():return {'model':'qwen/qwen3.8-27b','messages':[{'role':'system','content':SYSTEM},{'role':'user','content':'fixture'}],'tools':[ACT],'tool_choice':'required','provider':{},'max_tokens':1024,'reasoning':{'enabled':False}}
def act(a='DOWN',n=20):return response('act',{'action':a,'notes':'fixture note'},n)
def rows(path):return [json.loads(x) for x in path.read_text().splitlines()]

@pytest.mark.parametrize('fault',['429','malformed','missing_usage','wrong_provider','fallback','pipeline','reasoning','truncated','overrun','wrong_total','missing_rt'])
def test_http_failure_consumed_no_retry(tmp_path,fault):
 r=act()
 if fault=='429':r=(429,{'error':'rate limited'})
 if fault=='malformed':r=b'{'
 if fault=='missing_usage':r.pop('usage')
 if fault=='wrong_provider':r['provider']='Other'
 if fault=='fallback':r['openrouter_metadata']['attempt']=2
 if fault=='pipeline':r['openrouter_metadata']['pipeline']=['changed']
 if fault=='reasoning':r['choices'][0]['message']['reasoning_details']=DETAILS
 if fault=='truncated':r['choices'][0]['finish_reason']='length'
 if fault=='overrun':r['usage'].update(completion_tokens=1025,total_tokens=1125)
 if fault=='wrong_total':r['usage']['total_tokens']=2
 if fault=='missing_rt':r['usage'].pop('completion_tokens_details')
 ledger=tmp_path/'ledger.json';initial_ledger(ledger)
 with Server([r]) as s:
  t=Transport('control',tmp_path/'raw',ledger,time.monotonic()+1800,fixture_url=s.url)
  with pytest.raises(Stop):t(body())
  with pytest.raises(Stop):t(body())
  assert len(s.requests)==1
 l=read(ledger);assert l['stopped'] and l['arms']['control']['requests']==1
 assert (float(l['unknown_exposure_usd'])>0)==(fault in ['429','malformed','missing_usage','wrong_total'])
 receipt=read(tmp_path/'raw/001.receipt.json');assert receipt['latency_seconds']>=0
 assert (tmp_path/'raw/001.response.raw').exists()

@pytest.mark.parametrize('fault',['requests','prompt','completion','allocation','exposure','unknown','time','context'])
def test_pre_send_reservations(tmp_path,fault):
 ledger=tmp_path/'ledger.json';initial_ledger(ledger);l=read(ledger);b=body();deadline=time.monotonic()+1800
 if fault=='requests':l['arms']['control']['requests']=120
 if fault=='prompt':l['arms']['control']['prompt_tokens']=2400000
 if fault=='completion':l['arms']['control']['completion_tokens']=122880
 if fault=='allocation':l['arms']['control']['known_cost_usd']='0.826944'
 if fault=='exposure':l['known_cost_usd']='1.9'
 if fault=='unknown':l['unknown_exposure_usd']='0.1'
 if fault=='time':deadline=time.monotonic()+19
 if fault=='context':b['messages'][-1]['content']='x'*65536
 write(ledger,l)
 with Server([]) as s:
  t=Transport('control',tmp_path/'raw',ledger,deadline,fixture_url=s.url)
  with pytest.raises(Stop):t(b)
  assert not s.requests
 assert not list((tmp_path/'raw').glob('*.request.json'))


@pytest.fixture(scope='module')
def integrated(tmp_path_factory):
 p=tmp_path_factory.mktemp('actual-pair')/'run'
 replies=[response('retrieve',{'start_step':1,'count':2,'notes':'read prefix'}),response('query_model',{'action':'DOWN','notes':'compare'}),act('DOWN'),act('RIGHT'),act('DOWN'),act('LEFT')]
 with Server(replies) as server:r=run_pair(p,fixture_url=server.url,action_limit=2);requests=server.requests
 return p,r,requests

def test_actual_pair_frozen_import_query_binding_and_parity(integrated):
 from codec import unpack
 p,r,requests=integrated
 assert r['status']=='PAIR_FINISHED' and r['primary_levels']==[0,0]
 assert not (p/'constructor').exists()
 assert read(p/'CONSUMED.json')['order']==['treatment','control']
 assert r['imported_model']['freeze_sha256']==hash_file(HERE/'donor/FROZEN.json')
 assert hash_file(p/'control/child/INITIAL-EVIDENCE.json')==hash_file(p/'treatment/child/INITIAL-EVIDENCE.json')
 assert all(x['reasoning']=={'enabled':False,'exclude':False} and x['provider']==PROVIDER for x in requests)
 for req in requests:
  assert not any('index' in c for m in req['messages'] for c in m.get('tool_calls',[]))
 q=read(p/'treatment/child/QUERY-001.json');assert q['forced'] and q['action']=='RIGHT' and q['step']==8
 packet=unpack(json.loads(requests[0]['messages'][1]['content']))['packet']
 assert packet['predecision_query_results'][0]['binding_sha256']==hash_file(p/'treatment/child/QUERY-001.json')
 packet2=unpack(json.loads(requests[1]['messages'][1]['content']))
 assert len(packet2['previous_tool_result']['transitions'])==2
 join=read(p/'treatment/child/ACTION-JOIN-001.json');assert join['decision']['query_exposure_ids']==[1,2]
 bound=read(p/'treatment/child/ACTUAL-BOUND-008.json');assert bound['action']=='DOWN' and bound['query_ids']==[2]
 calls=rows(p/'treatment/child/model-calls.jsonl');assert all(c['args'][2]=={'status':'unscored'} for c in calls if c['method']=='assimilate')
 assert len(rows(p/'treatment/child/sealed-rpc.jsonl'))>20
 assert not (p/'treatment/child/QUERY-SCORE-001.json').exists()
 assert (p/'treatment/child/QUERY-SCORE-002.json').exists()
 for arm in ('control','treatment'):
  assert read(p/arm/'RECONCILIATION.json')['new_actions']==2
  assert read(p/arm/'child/WORKER.json')['credential_environment'] is False
 assert r['ledger']['unknown_exposure_usd']=='0'

def test_once_only(integrated):
 with pytest.raises(FileExistsError):run_pair(integrated[0],fixture_url='http://127.0.0.1:9/chat',action_limit=2)

def test_codec_prior_exact_requests(tmp_path):
 from codec import pack,unpack,messages
 from actor import RETRIEVE
 donor=Path(read(HERE/'DONOR.json')['donor']);results=[]
 for f in sorted((donor/'live/control/child').glob('ACTOR-INPUT-*.json')):
  b=read(f);packet=json.loads(b['messages'][1]['content']);exchange=b['messages'][2:]
  value={'packet':packet,'previous_tool_result':json.loads(exchange[-1]['content']) if exchange else None}
  assert unpack(pack(value))==value
  # Replay legacy packets losslessly; notes are NOT cut to the new bound in this audit.
  packet['executed_summary']=[[r['step'],r['action'],r['state'],r['levels_completed']] for r in packet['executed_summary']]
  b['messages']=[{'role':'system','content':SYSTEM}]+messages(packet,exchange);b['tools']=[ACT,RETRIEVE]
  reservation=len(json.dumps(wire(b,'control'),ensure_ascii=False,separators=(',',':')).encode())+4096
  assert reservation<=20000
  results.append({'sha256':hash_file(f),'decoded_equal':True,'new_reservation':reservation})
 assert len(results)==20
 put(tmp_path/'REPLAY.json',results)

def test_codec_general_integer_grid_and_dedup():
 from codec import pack,unpack
 v={'a':{'encoding':'integer_matrix','grid':[[100,-4,-4],[3,3,3]],'height':2,'width':3}}
 v['b']=copy.deepcopy(v['a']);p=pack(v)
 assert len(p['frames'])==1 and unpack(p)==v

def test_all240_maximum_usage_requests_admitted(tmp_path):
 from decimal import Decimal
 from transport import RATES
 b=body();base=len(json.dumps(wire(b,'control'),ensure_ascii=False,separators=(',',':')).encode())+4096
 b['messages'][-1]['content']+='x'*(20000-base)
 assert len(json.dumps(wire(b,'control'),ensure_ascii=False,separators=(',',':')).encode())+4096==20000
 nominal=20000*RATES['prompt']+1024*RATES['completion']
 replies=[]
 for i in range(240):
  r=act(n=i);r['usage'].update(prompt_tokens=20000,completion_tokens=1024,total_tokens=21024,cost=str(nominal));replies.append(r)
 ledger=tmp_path/'ledger.json';initial_ledger(ledger)
 with Server(replies) as s:
  for phase in ('treatment','control'):
   t=Transport(phase,tmp_path/phase,ledger,time.monotonic()+1800,fixture_url=s.url)
   for _ in range(120):t(b)
   with pytest.raises(Stop,match='budget'):t(b)
  assert len(s.requests)==240
 l=read(ledger);assert Decimal(l['known_cost_usd'])==Decimal('1.653888') and l['unknown_exposure_usd']=='0'
 assert all(a['prompt_tokens']==2400000 and a['completion_tokens']==122880 for a in l['arms'].values())
 put(tmp_path/'FULL-HORIZON.json',{'fixture':True,'ledger':l,'conservative_nominal_plus_extra_exposure':'1.9924016'})

@pytest.mark.parametrize('status',[200,429])
def test_request_absolute_deadline_success_and_error(tmp_path,status,monkeypatch):
 import http.server,threading,signal,deadline
 monkeypatch.setattr(deadline,'REQUEST_SECONDS',1.0)
 raw=encoded(act());sent=[]
 class Handler(http.server.BaseHTTPRequestHandler):
  def log_message(self,*args):pass
  def do_POST(self):
   self.rfile.read(int(self.headers['Content-Length']));self.send_response(status);self.send_header('Content-Length',str(len(raw)));self.end_headers()
   try:
    for i in range(0,len(raw),20):time.sleep(.05);self.wfile.write(raw[i:i+20]);self.wfile.flush();sent.append(i)
   except (BrokenPipeError,ConnectionResetError):pass
 server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
 ledger=tmp_path/'ledger.json';initial_ledger(ledger)
 t=Transport('treatment',tmp_path/'raw',ledger,time.monotonic()+1800,fixture_url=f'http://127.0.0.1:{server.server_port}/chat')
 root=tmp_path/'supervised'
 code='import pathlib,sys,time; p=pathlib.Path('+repr(str(root/'child'))+');p.mkdir();(p/"RESULT.json").write_text("{}");print('+repr(json.dumps({'request':body()}))+',flush=True);sys.stdin.readline();(p/"LATE_ACTION").write_text("forbidden");time.sleep(30)'
 previous=signal.getsignal(signal.SIGALRM)
 try:
  r=supervise('treatment',root,'-','-',t,hard=5,command=[sys.executable,'-B','-c',code])
  assert r['reaped'] and r['returncode']==-signal.SIGKILL and not r['hard_timeout']
  assert 1<=r['elapsed_seconds']<2.5 and 'request deadline' in r['failure']
  assert not (root/'child/LATE_ACTION').exists() and not promote(root,'treatment')
  rec=read(tmp_path/'raw/001.receipt.json');assert rec['error_type']=='RequestDeadline' and rec['http_status']==status and rec['response_complete'] is False
  assert read(ledger)['stopped'] and float(read(ledger)['unknown_exposure_usd'])>0
  with pytest.raises(Stop):t(body())
  assert signal.getsignal(signal.SIGALRM)==previous and signal.getitimer(signal.ITIMER_REAL)==(0.,0.)
  put(tmp_path/'DEADLINE.json',r)
 finally:server.shutdown();server.server_close();thread.join()

def test_timer_restores_remaining_phase_and_shorter_phase_wins(monkeypatch):
 import signal,deadline
 from run_utility import HardDeadline
 previous=signal.getsignal(signal.SIGALRM)
 def phase_expired(*a):raise HardDeadline()
 signal.signal(signal.SIGALRM,phase_expired);signal.setitimer(signal.ITIMER_REAL,1.)
 try:
  with deadline.request_deadline(time.monotonic()+1800):time.sleep(.05)
  remaining,_=signal.getitimer(signal.ITIMER_REAL);assert .8<remaining<.97
  signal.setitimer(signal.ITIMER_REAL,.05)
  with pytest.raises(HardDeadline):
   with deadline.request_deadline(time.monotonic()+1800):time.sleep(.2)
 finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
 assert deadline.REQUEST_SECONDS==90

def test_donor_drift_and_prefix_rejected(tmp_path):
 from run_utility import validate_donor
 assert validate_donor(HERE/'donor/prefix.jsonl')['source_digest']=='fd46b2b4db763deeca910161920adabf41dd3e4d1eceec3a14c7bd8e111fd107'
 f=tmp_path/'badprefix';f.write_text('{}')
 with pytest.raises(Stop,match='prefix'):validate_donor(f)

def test_unknown_stops_before_control(tmp_path):
 with Server([(429,{'error':'fixture'})]) as s:
  with pytest.raises(Stop):run_pair(tmp_path/'run',fixture_url=s.url,action_limit=1)
  assert len(s.requests)==1
 assert not (tmp_path/'run/control').exists()
 assert read(tmp_path/'run/RESULT.json')['status']=='HOLD'

@pytest.mark.parametrize('fault',['multiple_tools','future_retrieval','notes'])
def test_actor_fault_stops_without_second_arm(tmp_path,fault):
 r=act()
 if fault=='multiple_tools':r['choices'][0]['message']['tool_calls']*=2
 if fault=='future_retrieval':r=response('retrieve',{'start_step':9,'count':1,'notes':''})
 if fault=='notes':r=response('act',{'action':'DOWN','notes':'x'*513})
 with Server([r]) as s:
  with pytest.raises(Stop):run_pair(tmp_path/'run',fixture_url=s.url,action_limit=1)
 assert not (tmp_path/'run/control').exists()

def test_full_official_action_horizon(tmp_path):
 # Deterministic fixture repeats legal actions to actual terminal or full cap, not efficacy evidence.
 with Server([act('DOWN',i) for i in range(240)]) as s:r=run_pair(tmp_path/'run',fixture_url=s.url,action_limit=100)
 assert r['status']=='PAIR_FINISHED'
 for phase in ('treatment','control'):
  arm=r['arms'][phase];assert arm['new_actions']==100 or arm['observation']['state']!='NOT_FINISHED'
  assert arm['complete_exposure']
 assert [x['new_actions'] for x in r['matched_action_milestones']]==[20,40,60,80]
 assert max(a['reserved_prompt_tokens'] for a in r['ledger']['attempts'])<=20000

def test_full_history_query_context_stress_is_lossless(integrated,tmp_path):
 from codec import unpack,messages
 p,_,_=integrated;b=read(p/'treatment/child/ACTOR-INPUT-001.json')
 packet=unpack(json.loads(b['messages'][1]['content']))['packet']
 packet['notes']='𐀀'*64
 packet['executed_summary']=[[i,'CLICK '+str(i%64)+' '+str((i*7)%64),'NOT_FINISHED',0] for i in range(1,109)]
 q=packet['predecision_query_results'][0]
 packet['predecision_query_results']=[dict(q,id=i,binding_sha256=format(i,'064x'),time_ns=q['time_ns']+i) for i in range(1,21)]
 from protocol import assistant_message
 exchange=[assistant_message(response('query_model',{'action':'RIGHT','notes':packet['notes']})['choices'][0]['message']),{'role':'tool','tool_call_id':'fixture-1','content':json.dumps(packet['predecision_query_results'][-1])}]
 b['messages']=[{'role':'system','content':SYSTEM}]+messages(packet,exchange)
 assert unpack(json.loads(b['messages'][1]['content']))['packet']==packet
 reservation=len(json.dumps(wire(b,'treatment'),ensure_ascii=False,separators=(',',':')).encode())+4096
 assert reservation<=20000
 put(tmp_path/'CONTEXT-STRESS.json',{'reserved_prompt_tokens':reservation,'history_rows':108,'queries':20,'notes_unicode_characters':64,'decoded_equal':True})

def test_missing_review_gate_does_not_read_credentials(monkeypatch,tmp_path):
 import run_utility,auth
 monkeypatch.setattr(run_utility,'verify_inventory',lambda:'fixture_inventory')
 def forbidden(*a,**k):raise AssertionError('credential access')
 monkeypatch.setattr(auth,'load_key',forbidden)
 put(tmp_path/'HOLD.json',{'task':'WORLD-MODEL-UTILITY-RETRY','role':'independent-reviewer','decision':'HOLD','inventory_sha256':'fixture_inventory','endpoint':'reka/fp8'})
 with pytest.raises(Stop):run_utility.gates(tmp_path/'HOLD.json',tmp_path/'absent',tmp_path/'absent2')

def test_actual_imported_model_twenty_queries_retrieval_context(tmp_path):
 from codec import unpack,messages
 notes='𐀀'*64
 optional=[response('query_model',{'action':['UP','DOWN','LEFT','RIGHT'][i%4],'notes':notes},i+1) for i in range(19)]
 replies=optional+[response('retrieve',{'start_step':1,'count':2,'notes':notes},20),response('act',{'action':'DOWN','notes':notes},21),act('DOWN',22)]
 with Server(replies) as s:r=run_pair(tmp_path/'run',fixture_url=s.url,action_limit=1);requests=s.requests
 assert r['status']=='PAIR_FINISHED' and len(requests)==22
 arm=r['arms']['treatment'];assert arm['query_attempts']==arm['query_executions']==20
 child=tmp_path/'run/treatment/child'
 assert all(read(child/f'QUERY-{i:03}.json')['status']=='predicted' for i in range(1,21))
 b=copy.deepcopy(requests[20]);decoded=unpack(json.loads(b['messages'][1]['content']))
 assert len(decoded['packet']['predecision_query_results'])==20
 assert len(decoded['previous_tool_result']['transitions'])==2
 # Beyond reachable108-action history: explicit conservative120-row stress, never fed to live actors.
 packet=decoded['packet'];packet['executed_summary']=[[i,'CLICK '+str(i%64)+' '+str((i*7)%64),'NOT_FINISHED',0] for i in range(1,121)]
 exchange=copy.deepcopy(b['messages'][2:]);exchange[-1]['content']=json.dumps(decoded['previous_tool_result'])
 logical={**b,'provider':{},'reasoning':{'enabled':False}};logical.pop('stream')
 logical['messages']=b['messages'][:1]+messages(packet,exchange)
 assert unpack(json.loads(logical['messages'][1]['content']))=={'packet':packet,'previous_tool_result':decoded['previous_tool_result']}
 reservation=len(json.dumps(wire(logical,'treatment'),ensure_ascii=False,separators=(',',':')).encode())+4096
 measured=[a['reserved_prompt_tokens'] for a in r['ledger']['attempts'] if a['arm']=='treatment']
 put(tmp_path/'ACTUAL-TREATMENT-QUALIFICATION.json',{'source_digest':r['imported_model']['source_digest'],'first_request_reserved_prompt_tokens':measured[0],'maximum_actual_request_reservation':max(measured),'full120_history_with_actual20_queries_and_retrieval_reservation':reservation,'query_executions':20,'optional_queries':19,'retrieved_transitions':2,'notes_unicode_characters':64,'lossless':True,'paid':False})
 assert reservation<=20000
