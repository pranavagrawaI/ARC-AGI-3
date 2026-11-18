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

@pytest.fixture(scope='module')
def integrated(tmp_path_factory):
 p=tmp_path_factory.mktemp('actual-pair')/'run'
 replies=construction(repair=True)+[act(),act('LEFT')]+[response('retrieve',{'start_step':1,'count':2,'notes':'read old prefix'},30),response('query_model',{'action':'DOWN','notes':'compare another prediction'},31),act('DOWN',32),act('RIGHT',33)]
 with Server(replies) as server:r=run_pair(p,fixture_url=server.url,action_limit=2);requests=server.requests
 return p,r,requests

def test_real_pair_query_before_decision_and_actual_branch(integrated):
 p,r,requests=integrated
 assert r['status']=='PAIR_FINISHED' and r['interpretation']=='INCONCLUSIVE_TIED_PRIMARY'
 assert r['constructor']['frozen'] and r['primary_levels']==[0,0]
 assert hash_file(p/'control/child/INITIAL-EVIDENCE.json')==hash_file(p/'treatment/child/INITIAL-EVIDENCE.json')
 assert all(read(p/a/'child/PREFIX.json')==read(p/'prefix/child/PREFIX.json') for a in ['control','treatment'])
 assert len(rows(p/'control/child/transitions.jsonl'))==10
 constructor_req=requests[:3]
 assert all(x['reasoning']=={'enabled':True,'exclude':False,'effort':'medium'} for x in constructor_req)
 assert all(x['reasoning']=={'enabled':False,'exclude':False} for x in requests[3:])
 assert all(x['provider']==PROVIDER and 'parallel_tool_calls' not in x for x in requests)
 m=next(m for m in constructor_req[1]['messages'] if m['role']=='assistant')
 assert m['reasoning_details']==DETAILS and 'index' not in m['tool_calls'][0]
 assert 'work_item' in constructor_req[2]['messages'][-2]['content'] or 'work_item' in json.dumps(constructor_req[2])
 q=read(p/'treatment/child/QUERY-001.json');assert q['forced'] and q['action']=='RIGHT' and q['step']==8
 initial=read(p/'treatment/child/ACTOR-INPUT-001.json');packet=json.loads(initial['messages'][1]['content'])
 assert packet['predecision_query_results'][0]['binding_sha256']==hash_file(p/'treatment/child/QUERY-001.json')
 join=read(p/'treatment/child/ACTION-JOIN-001.json');assert join['transition']['action']=='DOWN'
 assert join['decision']['query_exposure_ids']==[1,2]
 assert join['decision']['input_sha256']==hash_file(p/'treatment/child/ACTOR-INPUT-003.json')
 bound=read(p/'treatment/child/ACTUAL-BOUND-008.json');assert bound['action']=='DOWN' and bound['query_ids']==[2]
 calls=rows(p/'treatment/child/model-calls.jsonl');ass=[x for x in calls if x['method']=='assimilate']
 assert ass[-2]['args'][0]['actual_action']=='DOWN' # discarded RIGHT state is never assimilated
 assert ass[-2]['args'][2]=={'status':'unscored'}
 assert not (p/'treatment/child/QUERY-SCORE-001.json').exists()
 assert (p/'treatment/child/QUERY-SCORE-002.json').exists()
 assert len(rows(p/'treatment/child/sealed-rpc.jsonl'))>20
 assert read(p/'treatment/child/WORKER.json')['credential_environment'] is False
 assert r['ledger']['unknown_exposure_usd']=='0'
 assert r['ledger']['arms']['constructor']['reasoning_tokens']>0
 assert r['ledger']['arms']['control']['reasoning_tokens']==r['ledger']['arms']['treatment']['reasoning_tokens']==0
 for a in ['control','treatment']:
  card=read(p/a/'child/SCORECARD.json');assert card['total_actions']==10 and card['total_levels_completed']==0

def test_once_only_pair(integrated):
 p,_,_=integrated
 with pytest.raises(FileExistsError):run_pair(p,fixture_url='http://127.0.0.1:9/chat',action_limit=2)

def test_constructor_failure_continues_without_oracle(tmp_path):
 replies=[response('request_evidence',{'obstacle':'unknown','needed_observation':'more data','decision_relevance':'unresolved'},phase='constructor'),act(),act()]
 with Server(replies) as s:r=run_pair(tmp_path/'run',fixture_url=s.url,action_limit=1)
 assert not r['constructor']['frozen'] and r['arms']['treatment']['new_actions']==1
 q=read(tmp_path/'run/treatment/child/QUERY-001.json');assert q['status']=='unavailable'
 assert not (tmp_path/'run/constructor/FROZEN.json').exists()

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
 if fault=='prompt':l['arms']['control']['prompt_tokens']=1500000
 if fault=='completion':l['arms']['control']['completion_tokens']=122880
 if fault=='allocation':l['arms']['control']['known_cost_usd']='0.634344'
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

@pytest.mark.parametrize('fault',['multiple_tools','future_retrieval','bad_notes'])
def test_actual_actor_failure_stops_pair(tmp_path,fault):
 r=act()
 if fault=='multiple_tools':r['choices'][0]['message']['tool_calls']*=2
 if fault=='future_retrieval':r=response('retrieve',{'start_step':9,'count':1,'notes':''})
 if fault=='bad_notes':r= response('act',{'action':'RIGHT','notes':'x'*2001})
 replies=[response('request_evidence',{'obstacle':'x','needed_observation':'x','decision_relevance':'x'},phase='constructor'),r]
 with Server(replies) as s:
  with pytest.raises(Stop):run_pair(tmp_path/'run',fixture_url=s.url,action_limit=1)
  assert len(s.requests)==2
 assert not (tmp_path/'run/treatment').exists()
 assert len(rows(tmp_path/'run/control/child/transitions.jsonl'))==8

@pytest.mark.parametrize('source',[SOURCE.replace('return {"grid":observation["grid"]}', 'return {"secret":open("/etc/passwd").read()}',1),SOURCE.replace('return {"grid":observation["grid"]}', 'import os\n os.system("true")\n return {}',1)])
def test_malicious_source_real_constructor_rejected(tmp_path,source):
 replies=construction(source)+[response('request_evidence',{'obstacle':'rejected','needed_observation':'none','decision_relevance':'fixture'},3,'constructor'),act(),act()]
 with Server(replies) as s:r=run_pair(tmp_path/'run',fixture_url=s.url,action_limit=1)
 assert not r['constructor']['frozen']
 events=rows(tmp_path/'run/constructor/child/tool-events.jsonl')
 assert 'failed' in json.dumps(events[1])
 assert r['arms']['treatment']['query_attempts']==1 and r['arms']['treatment']['query_executions']==0


def test_post_child_write_hard_kill_no_promotion(tmp_path):
 root=tmp_path/'supervised';code='import pathlib,time; p=pathlib.Path('+repr(str(root/'child'))+');p.mkdir();(p/"FROZEN.json").write_text("{}");(p/"RESULT.json").write_text(\'{"frozen":true}\');time.sleep(10)'
 receipt=supervise('constructor',root,'-','-',hard=.2,command=[sys.executable,'-B','-c',code])
 assert receipt['hard_timeout'] and receipt['reaped'] and (root/'child/FROZEN.json').exists()
 assert not promote(root,'constructor') and not (root/'FROZEN.json').exists()


def test_public_discovery_pin_and_transport_optional_fields():
 d=read(HERE/'discovery/RECEIPT.json')['selected']
 assert d['tag']=='reka/fp8' and d['pricing']['prompt']=='0.000000214' and d['pricing']['completion']=='0.00000255'
 assert d['max_completion_tokens']==131072 and d['context_length']==262144
 assert {'reasoning','max_tokens','tools','tool_choice'}<=set(d['supported_parameters'])
 assert set(wire(body(),'control'))=={'model','messages','provider','tools','tool_choice','max_tokens','reasoning','stream'}

@pytest.mark.parametrize('extra',[False,True])
def test_actual_query_limit_null_state_abstain(tmp_path,extra):
 source='''def initialize(observation):
 return {"grid":observation["grid"]}
def predict(state,action):
 return [{"state":None,"commitments":{"type":"abstain","reason":"fixture uncertainty"}}]
def assimilate(predicted_state,observation,verdict):
 assert predicted_state is None
 return {"grid":observation["grid"]}
'''
 replies=construction(source)+[act()]+[response('query_model',{'action':'RIGHT','notes':'fixture'},40+i) for i in range(19+int(extra))]+[act('RIGHT',70)]
 with Server(replies) as s:
  if extra:
   with pytest.raises(Stop):run_pair(tmp_path/'run',fixture_url=s.url,action_limit=1)
  else:r=run_pair(tmp_path/'run',fixture_url=s.url,action_limit=1);assert r['status']=='PAIR_FINISHED'
 child=tmp_path/'run/treatment/child'
 assert len(list(child.glob('QUERY-[0-9]*.json')))==20
 assert all(read(q)['value'][0]['commitments']['type']=='abstain' for q in child.glob('QUERY-[0-9]*.json'))
 if not extra:
  assert read(child/'ACTUAL-BOUND-008.json')['value'][0]['state'] is None
  assert read(child/'QUERY-SCORE-001.json')['metrics']['coverage']==0


def test_unknown_constructor_attempt_halts_before_actors(tmp_path):
 with Server([(429,{'error':'fixture'})]) as s:
  with pytest.raises(Stop):run_pair(tmp_path/'run',fixture_url=s.url,action_limit=1)
  assert len(s.requests)==1
 assert not (tmp_path/'run/control').exists() and not (tmp_path/'run/constructor/FROZEN.json').exists()
 assert float(read(tmp_path/'run/ledger.json')['unknown_exposure_usd'])>0


def test_prefix_mismatch_prevents_actor_request(tmp_path):
 r=supervise('prefix',tmp_path/'prefix','-','-');assert r['returncode']==0
 path=tmp_path/'prefix/child/PREFIX.json';d=read(path);d['sha256']='wrong';path.write_bytes(encoded(d))
 with Server([]) as s:
  ledger=tmp_path/'ledger.json';initial_ledger(ledger)
  t=Transport('control',tmp_path/'raw',ledger,time.monotonic()+1800,fixture_url=s.url)
  r=supervise('control',tmp_path/'actor',tmp_path/'prefix/child/prefix.jsonl','-',t,limit=1)
  assert r['returncode']!=0 and not s.requests


def test_level_and_reset_disable_under_real_sealed_rpc(tmp_path):
 # Calls Model's actual sealed boundary in a mapped namespace, not an imitation RPC.
 code='''import sys,json
from pathlib import Path
p=Path(sys.argv[1]);sys.path[:0]=[str(p),str(p/'engine')]
from model import Model,audit
from common import put,encoded,digest
out=Path(sys.argv[2]);out.mkdir()
source=%s
obs={'grid':[[0]],'state':'NOT_FINISHED','levels_completed':0,'available_actions':['RIGHT','RESET']}
row={'step':1,'turn':0,'action':'RIGHT','before':obs,'after':obs}
prefix=out/'prefix.jsonl';prefix.write_bytes(encoded(row))
f=out/'FROZEN.json';put(f,{'candidate_source':source,'source_digest':digest(source.encode()),'prefix_sha256':digest(prefix.read_bytes())})
with audit(out):
 m=Model(f,prefix,out);assert m.active
 m.query('RIGHT',obs,1);m.before_actual('RIGHT',obs,1)
 m.after_actual({'step':2,'action':'RIGHT','before':obs,'after':dict(obs,levels_completed=1)})
 assert not m.active and m.reason=='level/terminal boundary'
 m=Model(f,prefix,out);m.before_actual('RESET',obs,1)
 assert not m.active and m.reason=='reset committed'
''' % repr(SOURCE)
 import subprocess
 script=tmp_path/'probe.py';script.write_text(code)
 r=subprocess.run(['unshare','--user','--map-auto','--map-root-user','--net',sys.executable,'-B',str(script),str(HERE),str(tmp_path/'model')],capture_output=True,text=True)
 assert r.returncode==0,r.stderr


def test_gate_failures_without_auth(monkeypatch,tmp_path):
 import run_utility
 monkeypatch.setattr(run_utility,'verify_inventory',lambda:'bound')
 with pytest.raises(Stop,match='absolute'):run_utility.gates(Path('relative'),Path('relative'),Path('relative'))
 # In-memory fixtures only; never write independent approval/release documents.
 data={'task':'WORLD-MODEL-UTILITY','role':'independent-reviewer','decision':'HOLD','inventory_sha256':'bound','endpoint':'reka/fp8'}
 monkeypatch.setattr(run_utility,'read',lambda p:data)
 with pytest.raises(Stop,match='gate'):run_utility.gates(tmp_path/'review',tmp_path/'release',tmp_path/'approval')


def test_full100_action_ceiling_real_official_game(tmp_path):
 # Fixed manufactured alternating directions, no search or outcome-driven script tuning.
 responses=construction()+[act('RIGHT' if i%2==0 else 'LEFT',100+i) for i in range(200)]
 with Server(responses) as s:r=run_pair(tmp_path/'run',fixture_url=s.url,action_limit=100)
 assert r['status']=='PAIR_FINISHED'
 for phase in ['control','treatment']:
  result=r['arms'][phase]
  assert result['new_actions']==100 or result['observation']['state']!='NOT_FINISHED'
  assert result['requests']<=100
  assert read(tmp_path/'run'/phase/'RECONCILIATION.json')['actions_including_prefix']==8+result['new_actions']
 assert all(a['reserved_prompt_tokens']<=65536 for a in r['ledger']['attempts'] if a['arm']!='constructor')


def test_actual120_request_cap_and_symmetric_allocation(tmp_path):
 ledger=tmp_path/'ledger.json';initial_ledger(ledger)
 with Server([act(n=i) for i in range(120)]) as s:
  t=Transport('control',tmp_path/'raw',ledger,time.monotonic()+1800,fixture_url=s.url)
  for i in range(120):t(body())
  with pytest.raises(Stop,match='budget'):t(body())
  assert len(s.requests)==120
 state=read(ledger);assert state['arms']['control']['requests']==120 and state['arms']['treatment']['requests']==0
 assert CAPS['control']['allocation']==CAPS['treatment']['allocation']
