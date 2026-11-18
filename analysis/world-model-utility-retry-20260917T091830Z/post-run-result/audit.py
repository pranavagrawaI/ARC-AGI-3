"""Read-only audit of once-only run; writes new result artifacts here, no remote calls."""
import json,hashlib,sys,os,shlex
from pathlib import Path
from decimal import Decimal
OUT=Path(__file__).resolve().parent;P=OUT.parent;LIVE=P/'live'
sys.path[:0]=[str(P),str(P/'engine')]
from codec import unpack
from run_utility import verify_inventory
read=lambda p:json.loads(p.read_bytes())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):
 with (OUT/n).open('x') as f:json.dump(v,f,indent=2,sort_keys=True);f.write('\n')
ledger=read(LIVE/'ledger.json');known=Decimal(0);receipts=[]
for a in ledger['attempts']:
 stem=LIVE/(a['arm']+'-raw')/f"{a['ordinal']:03}"
 request=stem.with_suffix('.request.json');response=stem.with_suffix('.response.raw');receipt=stem.with_suffix('.receipt.json')
 assert sha(request)==a['request_sha256'] and sha(response)==a['response_sha256']
 raw=read(response);rec=read(receipt);wire=read(request)
 assert rec==a and raw['usage']==a['usage'] and a['status']=='receipted'
 assert wire['reasoning']=={'enabled':False,'exclude':False} and wire['provider']['only']==['reka/fp8'] and not wire['provider']['allow_fallbacks']
 assert raw['provider']=='Reka' and raw['openrouter_metadata']['attempt']==1
 assert raw['usage']['completion_tokens_details']['reasoning_tokens']==0 and not a['reasoning_payload_present']
 cost=Decimal(str(raw['usage']['cost']));assert cost==Decimal(a['actual_cost_usd']);known+=cost
 receipts.append({'ordinal':a['ordinal'],'request_sha256':sha(request),'response_sha256':sha(response),'receipt_sha256':sha(receipt),'cost_usd':str(cost),'prompt_tokens':raw['usage']['prompt_tokens'],'completion_tokens':raw['usage']['completion_tokens'],'latency_seconds':a['latency_seconds'],'reserved_prompt_tokens':a['reserved_prompt_tokens']})
assert known==Decimal(ledger['known_cost_usd']) and ledger['unknown_exposure_usd']=='0'
child=LIVE/'treatment/child';actor=read(child/'ACTOR-RESULT.json');rows=[json.loads(x) for x in (child/'transitions.jsonl').read_text().splitlines()]
assert len(rows)==8+actor['new_actions'] and rows[-1]['after']==actor['observation']
assert all(a['after']==b['before'] for a,b in zip(rows,rows[1:]))
sc=read(child/'SCORECARD.json');run=sc['environments'][0]['runs'][0]
assert run['actions']==len(rows) and run['levels_completed']==actor['observation']['levels_completed'] and run['state']==actor['observation']['state']
joins=[]
for f in sorted(child.glob('ACTION-JOIN-*.json')):
 j=read(f);d=j['decision'];n=d['call'];inp=child/f'ACTOR-INPUT-{n:03}.json';resp=child/f'ACTOR-RESPONSE-{n:03}.json'
 assert sha(inp)==d['input_sha256'] and sha(resp)==d['response_sha256']
 packet=unpack(json.loads(read(inp)['messages'][1]['content']))['packet']
 assert packet['predecision_query_results']==j['predecision_queries']
 for q in j['predecision_queries']:assert q['binding_sha256']==sha(child/f"QUERY-{q['id']:03}.json")
 joins.append({'call':n,'action':j['transition']['action'],'step':j['transition']['step'],'query_exposure_ids':d['query_exposure_ids']})
last=read(child/'ACTOR-RESPONSE-010.json')['choices'][0]['message']['tool_calls'][0];args=json.loads(last['function']['arguments'])
assert last['function']['name']=='retrieve' and args['start_step']==14 and len(rows)==13
parent=int(read(LIVE/'treatment/COMMAND.json')['environment']['UTILITY_PARENT_PID']);assert not Path(f'/proc/{parent}').exists()
owned=[]
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
 try:
  argv=(proc/'cmdline').read_bytes().split(b'\0')
  if any(a==str(P/n).encode() for a in argv for n in ('worker.py','run_utility.py')) or str((proc/'cwd').resolve()).startswith(str(LIVE)):owned.append(int(proc.name))
 except OSError:pass
assert not owned
inventory=verify_inventory();donor=read(P/'DONOR.json');old=Path(donor['donor'])
assert sha(old/'live/RESULT.json')==donor['old_result_sha256'] and sha(old/'INVENTORY.json')==donor['old_inventory_sha256']
summary={'task':'WORLD-MODEL-UTILITY-RETRY','status':'HOLD','execution_count':1,'command_exit_code':2,'inventory_sha256':inventory,'stop':read(child/'ACTOR-FAILURE.json'),'rejected_tool':{'name':last['function']['name'],'arguments':args,'executed_transitions_available':len(rows)},'known_cost_usd':str(known),'unresolved_exposure_usd':ledger['unknown_exposure_usd'],'requests':len(receipts),'usage':ledger['arms'],'treatment':{k:v for k,v in actor.items() if k!='observation'},'official_outcome':{'new_actions':actor['new_actions'],'actions_including_prefix':run['actions'],'levels_completed':run['levels_completed'],'state':run['state'],'score':run['score'],'scorecard_closed':True,'root_result_promoted':(LIVE/'treatment/RESULT.json').exists()},'control':{'started':False,'outcome':None},'action_exposures':joins,'matched_action_milestones':[],'supervision':read(LIVE/'treatment/SUPERVISION.json'),'parent_pid':parent,'parent_absent':True,'owned_workers_remaining':owned,'old_run_preserved':True,'old_known_cost_usd':donor['old_known_cost_usd'],'old_unresolved_exposure_usd':donor['old_unknown_exposure_usd'],'interpretation':'COMPARISON_NOT_COMPLETED; no scientific inference; independent read-only outcome audit pending','allowed_next_action':'read-only audit only; no retry, fallback or source repair'}
save('SUMMARY.json',summary);save('RECEIPT-AUDIT.json',receipts)
command=P.joinpath('README.md').read_text().split('## EXACT LIVE COMMAND')[1].split('```bash\n')[1].split('\n```')[0]
save('EXECUTION.json',{'command':command,'argv':shlex.split(command),'tool_session_id':60950,'exit_code':2,'once_only':True,'gate_hashes':{n:sha(P/n) for n in ['INVENTORY.json','independent-review/DECISION.json','HERMES-RELEASE.json','PM-CREDENTIAL-APPROVAL.json']}})
save('EVIDENCE-HASHES.json',{str(f.relative_to(P)):sha(f) for f in sorted(LIVE.rglob('*')) if f.is_file() and not f.is_symlink()})
print(json.dumps({'status':summary['status'],'known_cost_usd':str(known),'unknown_exposure_usd':ledger['unknown_exposure_usd'],'requests':len(receipts),'official_outcome':summary['official_outcome'],'query_attempts':actor['query_attempts'],'query_executions':actor['query_executions'],'query_scores':len(list(child.glob('QUERY-SCORE-*.json'))),'max_context_reservation':max(r['reserved_prompt_tokens'] for r in receipts),'parent_absent':True}))
