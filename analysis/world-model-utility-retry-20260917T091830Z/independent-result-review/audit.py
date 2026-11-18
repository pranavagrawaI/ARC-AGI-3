"""Offline preserved-result audit. No provider, auth, game stepping or candidate-source execution.
Reproduce only the actor retrieval branch with a historical-state adapter and raw saved response.
Writes fresh local boundary evidence under this script's directory; prints summary JSON.
"""
import sys,json,hashlib,copy,tempfile
from decimal import Decimal
from pathlib import Path
P=Path(__file__).resolve().parents[1];O=Path(__file__).resolve().parent
sys.path[:0]=[str(P),str(P/'engine')]
from codec import unpack
H=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
J=lambda f:json.loads(f.read_bytes())
L=P/'live';C=L/'treatment/child';inv=J(P/'INVENTORY.json')
assert H(P/'INVENTORY.json')=='6222b21ed809bd4e1a41043593e4a594b9390f7a69a18d4dad1dc5213bdeafc7'
for n,h in inv['files'].items():assert H(P/n)==h,n
for n,h in inv['runtime'].items():assert H(Path(n))==h,n
assert J(P/'independent-review/DECISION.json')['inventory_sha256']==H(P/'INVENTORY.json')
ledger=J(L/'ledger.json');cost=Decimal(0);pt=ct=rt=0;calls=[];payloads=[]
assert len(ledger['attempts'])==10
for i in range(1,11):
 base=L/'treatment-raw'/f'{i:03}';req=J(base.with_suffix('.request.json'));res=J(base.with_suffix('.response.raw'));rec=J(base.with_suffix('.receipt.json'))
 assert rec['request_sha256']==H(base.with_suffix('.request.json')) and rec['response_sha256']==H(base.with_suffix('.response.raw'))
 assert rec['status']=='receipted' and rec['http_status']==200 and not rec['error_type']
 u=res['usage'];assert u==rec['usage'];cost+=Decimal(str(u['cost']));pt+=u['prompt_tokens'];ct+=u['completion_tokens'];rt+=u['completion_tokens_details']['reasoning_tokens']
 assert req['reasoning']=={'enabled':False,'exclude':False} and req['provider']['only']==['reka/fp8'] and not req['provider']['allow_fallbacks']
 assert res['provider']=='Reka' and res['openrouter_metadata']['attempt']==1
 assert res['choices'][0]['finish_reason']=='tool_calls'
 m=res['choices'][0]['message'];assert not m.get('reasoning') and not m.get('reasoning_details')
 child_input=J(C/f'ACTOR-INPUT-{i:03}.json');assert child_input['messages']==req['messages'] and child_input['tools']==req['tools']
 assert J(C/f'ACTOR-RESPONSE-{i:03}.json')==res
 tc=m['tool_calls'][0]['function'];args=json.loads(tc['arguments']);d=J(C/f'DECISION-{i:03}.json')
 assert d['arguments']==args and d['tool']==tc['name'] and d['input_sha256']==H(C/f'ACTOR-INPUT-{i:03}.json')
 calls.append({'request':i,'tool':tc['name'],'arguments':args,'step_before':d['step_before']})
 payloads.append(unpack(json.loads(req['messages'][1]['content'])))
assert cost==Decimal('0.006547858')==Decimal(ledger['known_cost_usd']) and ledger['unknown_exposure_usd']=='0'
assert (pt,ct,rt)==(24997,470,0)
assert ledger['arms']['treatment']['requests']==10 and ledger['arms']['control']['requests']==0 and not (L/'control').exists()
rows=[json.loads(x) for x in (C/'transitions.jsonl').read_text().splitlines()];assert len(rows)==13
assert [r['step'] for r in rows]==list(range(1,14))
assert all(a['after']==b['before'] for a,b in zip(rows,rows[1:]))
result=J(C/'ACTOR-RESULT.json');assert result['new_actions']==5 and result['observation']==rows[-1]['after']
assert result['observation']['levels_completed']==0 and result['observation']['state']=='NOT_FINISHED'
card=J(C/'SCORECARD.json');run=card['environments'][0]['runs'][0]
assert run['actions']==13 and run['levels_completed']==0 and run['state']=='NOT_FINISHED'
last=calls[-1];assert last['arguments']=={'count':1,'notes':'verify step14 state','start_step':14} and last['step_before']==13
assert [r[0] for r in payloads[-1]['packet']['executed_summary']]==list(range(1,14))
assert payloads[-1]['packet']['new_actions']==5 and payloads[-1]['packet']['recent_transitions'][-1]['step']==13
scores=[]
for qid,action_step in [(1,9),(2,9),(3,10)]:
 q=J(C/f'QUERY-{qid:03}.json');s=J(C/f'QUERY-SCORE-{qid:03}.json');row=rows[action_step-1]
 assert s['query_sha256']==H(C/f'QUERY-{qid:03}.json') and q['step']+1==action_step and q['action']==row['action']
 a=[v for r in row['before']['grid'] for v in r];y=[v for r in row['after']['grid'] for v in r];z=[v for r in q['value'][0]['commitments']['predicted_grid'] for v in r]
 assert z==a
 changed=sum(x!=v for x,v in zip(a,y));acc=sum(x==v for x,v in zip(z,y))/len(y)
 assert s['metrics']['changed_cells']==changed and s['metrics']['prediction']['cell_accuracy']==acc and s['metrics']['paired_cell_accuracy_difference']==0
 scores.append({'query':qid,'action_step':action_step,'changed_cells':changed,'exact':z==y,'cell_accuracy':acc,'improvement_over_identity':0})
for i in range(1,6):
 join=J(C/f'ACTION-JOIN-{i:03}.json');d=join['decision'];packet=payloads[d['call']-1]['packet']
 assert join['transition']==rows[i+7] and packet['predecision_query_results']==join['predecision_queries']
 for q in join['predecision_queries']:assert q['binding_sha256']==H(C/f'QUERY-{q["id"]:03}.json')
assert payloads[0]['packet']['predecision_query_results'][0]['id']==1
assert [q['id'] for q in payloads[1]['packet']['predecision_query_results']]==[1,2]
assert [q['id'] for q in payloads[4]['packet']['predecision_query_results']]==[3]
# Real host actor.run retrieval validation; no actual game or model calls.
from actor import run as actor_run
class HistoricalGame:
 def __init__(self):self.rows=copy.deepcopy(rows);self.current=copy.deepcopy(rows[-1]['after'])
 def prefix(self):return 'historical-fixture'
 def step(self,*a):raise AssertionError('must not step game')
out=Path(tempfile.mkdtemp(prefix='retrieval-boundary-',dir=O))
raw10=J(L/'treatment-raw/010.response.raw')
def recorded_response(body):return copy.deepcopy(raw10)
try:
 actor_run(HistoricalGame(),'control',None,recorded_response,out,'historical-fixture',action_limit=100)
 raise AssertionError('expected retrieval rejection')
except ValueError as e:assert str(e)=='retrieval future/range'
assert J(out/'ACTOR-FAILURE.json')==J(C/'ACTOR-FAILURE.json')
assert not list(out.glob('ACTION-JOIN-*'))
old=Path(J(P/'DONOR.json')['donor']);oldinv=J(old/'INVENTORY.json')
assert H(old/'INVENTORY.json')=='ac86862b5de9ddb3598205580417655ce3349ba3bc9309ee793e9b15892ea71b'
for n,h in oldinv['files'].items():assert H(old/n)==h,n
ol=J(old/'live/ledger.json');assert ol['known_cost_usd']=='0.170458010' and ol['unknown_exposure_usd']=='0.345637232' and ol['stopped']
assert H(old/'live/ledger.json')==J(P/'independent-review/bindings.json')['old_ledger_sha256']
assert J(L/'RESULT.json')['status']=='HOLD' and not (L/'treatment/RESULT.json').exists()
summary={'status':'AUDITED_INCOMPLETE_NO_UTILITY_CONCLUSION','inventory_sha256':H(P/'INVENTORY.json'),'known_cost_usd':str(cost),'new_unknown_exposure_usd':'0','prompt_tokens':pt,'completion_tokens':ct,'reasoning_tokens':rt,'calls':calls,'actions':[r['action'] for r in rows[8:]],'official_levels':0,'official_state':'NOT_FINISHED','official_score':run['score'],'control_absent':True,'query_scores':scores,'unique_scored_transitions':2,'retrieval_boundary_reproduced':True,'boundary_evidence':str(out.relative_to(O)),'supervision':J(L/'treatment/SUPERVISION.json'),'old_unknown_exposure_usd':ol['unknown_exposure_usd'],'input_hashes':{str(f.relative_to(P)):H(f) for f in L.rglob('*') if f.is_file() and '__pycache__' not in f.parts},'review_decision_sha256':H(P/'independent-review/DECISION.json')}
print(json.dumps(summary,indent=2))
