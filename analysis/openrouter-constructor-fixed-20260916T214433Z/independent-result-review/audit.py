"""Offline reconciliation; never imports or executes candidate code."""
import ast,copy,hashlib,json
from decimal import Decimal as D
from pathlib import Path
O=Path(__file__).resolve().parent;B=O.parent;L=B/'live';seen={}
def read(p):
 b=p.read_bytes();seen[str(p)]=hashlib.sha256(b).hexdigest();return b
def js(p):return json.loads(read(p))
def rows(p):return [json.loads(x) for x in read(p).splitlines() if x.strip()]
def sha(b):return hashlib.sha256(b).hexdigest()
def argsha(args):return sha(json.dumps(args,sort_keys=True,separators=(',',':')).encode())
inv=js(B/'INVENTORY.json');ih=sha(read(B/'INVENTORY.json'));assert ih=='bba4d707c6c5c6edae7e818ca68ed8b0c969066832509b4acd2500f29b16276a'
for n,h in inv['files'].items():assert sha(read(B/n))==h
snap=js(B/'PRIOR-EXPERIMENT-HASHES.json');OLD=Path(snap['root'])
for n,h in snap['files'].items():assert sha(read(OLD/n))==h
manifest=B.parent/'r10-prep-20260916/r10a-cd82-20260916/manifest.json';m=js(manifest)
for n,h in m['sources'].items():assert sha(read(Path(n)))==h
# Bind the review decision/report and retained review-time evidence too.
for p in (B/'independent-review').iterdir():
 if p.is_file():read(p)
review_hashes=js(B/'independent-review/EVIDENCE-HASHES.json')
for n,h in review_hashes.items():assert sha(read(B/'independent-review'/n))==h
ledger=js(L/'ledger.json');pair=js(L/'RESULT.json');assert pair['ledger']==ledger and pair['inventory_sha256']==ih
assert not ledger['stopped'] and D(ledger['unknown_exposure_usd'])==0
expected=[('off',i) for i in range(1,6)]+[('on',i) for i in range(1,5)]
assert [(r['arm'],r['ordinal']) for r in ledger['attempts']]==expected
prefix=js(B/'prefix.jsonl');assert prefix['step']==1 and prefix['action']=='RIGHT'
result={'inventory_sha256':ih,'prior_files_unchanged':len(snap['files']),'r10_sources_unchanged':len(m['sources']),'arms':{}};first={};cumulative=D(0)
for arm,n in [('off',5),('on',4)]:
 C=L/arm/'child';root=js(L/arm/'RESULT.json');child=js(C/'RESULT.json');sup=js(L/arm/'SUPERVISION.json');f=js(L/arm/'FROZEN.json');assert read(L/arm/'FROZEN.json')==read(C/'FROZEN.json')
 assert {k:v for k,v in root.items() if k!='authority'}==child and root['authority']=='timely root promotion'
 assert root['frozen'] and root['construction_status']=='CANDIDATE_READY' and not root['supported_model'] and root['prospective']=='not_run'
 assert sup['returncode']==0 and not sup['hard_timeout'] and sup['reaped'] and sup['elapsed_seconds']<245 and root['elapsed_seconds']<240
 progress=js(C/'PROGRESS.json');events=rows(C/'tool-events.jsonl');assert events==progress['tool_events'];assert len(events)==len(progress['transcript'])==n
 totals={'requests':n,'prompt_tokens':0,'completion_tokens':0,'reasoning_tokens':0};cost=D(0);cached=0;feedback=[];assistants=[];calls=[]
 for i in range(1,n+1):
  qb=read(C/f'raw/{i:03}.request.json');rb=read(C/f'raw/{i:03}.response.raw');q=json.loads(qb);r=json.loads(rb);rec=js(C/f'raw/{i:03}.receipt.json');assert rec==next(x for x in ledger['attempts'] if x['arm']==arm and x['ordinal']==i)
  assert sha(qb)==rec['request_sha256'] and sha(rb)==rec['response_sha256'];assert rec['http_status']==200 and rec['error_type'] is None
  assert progress['transcript'][i-1]['response']==r
  logical=copy.deepcopy(progress['transcript'][i-1]['request']);logical.update(provider=q['provider'],reasoning=q['reasoning'],stream=False);assert logical==q
  assert q['provider']['order']==q['provider']['only']==['reka/fp8'] and not q['provider']['allow_fallbacks'] and q['provider']['require_parameters']
  assert q['reasoning']==({'enabled':False,'exclude':False} if arm=='off' else {'enabled':True,'exclude':False,'effort':'medium'})
  if i==1:first[arm]=copy.deepcopy(q);first[arm].pop('reasoning')
  u=r['usage'];assert u==rec['usage'];rt=u['completion_tokens_details']['reasoning_tokens'];assert 0<=rt<=u['completion_tokens']<=q['max_tokens']==4096
  assert u['total_tokens']==u['prompt_tokens']+u['completion_tokens'];assert u['prompt_tokens']<=rec['reserved_prompt_tokens']
  for k in ['prompt_tokens','completion_tokens']:totals[k]+=u[k]
  totals['reasoning_tokens']+=rt;cached+=u['prompt_tokens_details']['cached_tokens'];charge=D(str(u['cost']));cost+=charge
  calc=D(u['prompt_tokens']-u['prompt_tokens_details']['cached_tokens'])*D('.000000214')+D(u['prompt_tokens_details']['cached_tokens'])*D('.00000015')+D(u['completion_tokens'])*D('.00000255');assert calc==charge==D(rec['actual_cost_usd'])
  exposure=D(len(qb)+4096)*D('.000000214')+D(131072)*D('.00000255');assert exposure==D(rec['exposure_usd']) and cumulative+exposure<=2;cumulative+=charge
  meta=r['openrouter_metadata'];assert r['provider']=='Reka' and meta['attempt']==1 and meta['strategy']=='direct' and not meta.get('pipeline');assert [e for e in meta['endpoints']['available'] if e.get('selected')]==[{'provider':'Reka','model':'qwen/qwen3.8-27b-20260814','selected':True}]
  msg=r['choices'][0]['message'];assert r['choices'][0]['finish_reason']=='tool_calls';assert (rt>0)==bool(msg.get('reasoning_details') or msg.get('reasoning'))==(arm=='on')
  assert [x for x in q['messages'] if x['role']=='tool']==feedback;assert [x for x in q['messages'] if x['role']=='assistant']==assistants
  proj={'role':'assistant','content':msg.get('content'),'tool_calls':[{'id':x['id'],'type':x['type'],'function':{k:x['function'][k] for k in ['name','arguments']}} for x in msg['tool_calls']]}
  if msg.get('reasoning_details'):proj['reasoning_details']=msg['reasoning_details']
  elif msg.get('reasoning'):proj['reasoning']=msg['reasoning']
  assistants.append(proj);e=events[i-1];assert msg['tool_calls']==[e['call']];feedback.append(e['rendered_result_message']);args=json.loads(e['call']['function']['arguments']);payload=json.loads(e['rendered_result_message']['content']);name=e['call']['function']['name']
  calls.append({'ordinal':i,'tool':name,'reasoning_tokens':rt,'completion_tokens':u['completion_tokens'],'cost_usd':str(charge)})
  if i<n:assert name=='execute_python' and payload['status']=='completed' and payload['returncode']==0
  else:
   assert name=='submit_candidate' and payload['status']=='evaluated';assert isinstance(args['work_item'],dict) and isinstance(args['candidate_source'],list) and all(isinstance(x,str) for x in args['candidate_source'])
   source='\n'.join(args['candidate_source'])+'\n';assert source==f['candidate_source']==progress['candidate_source'];assert sha(source.encode())==f['source_digest']==progress['evaluation']['model_sha256']
   assert args['work_item']==progress['work_item'];schema=next(t['function']['parameters']['properties']['work_item'] for t in q['tools'] if t['function']['name']=='submit_candidate');assert set(schema['required'])==set(args['work_item']) and not schema['additionalProperties']
   assert args['work_item']['evidence_refs']==args['work_item']['completion_checks']==[1]
   ev=progress['evaluation'];assert payload['visible_evaluation']=={k:v for k,v in ev.items() if k!='suffix'}
   assert ev['visible']=={'status':'passed','kind':'visible_abi_replay','checks':[{'step':1,'action':'RIGHT','status':'abi_replayed'}]}
   assert ev['coverage']=={'required':[],'missing_required':[]} and not ev['dependencies']['static_completeness_established'];assert ev['evidence_claim']==f['evidence_claim']=='ABI_VISIBLE_REPLAY_ONLY_NOT_SUPPORTED_MODEL'
   # Independent manual state derivation from inspected short sources, no execution.
   initial={'grid':prefix['before']['grid'],'step':0};predicted={'grid':prefix['before']['grid'],'step':1}
   if arm=='on':initial['last_action']=None;predicted['last_action']='RIGHT'
   expected_args=[[{'grid':prefix['before']['grid']}],[initial,'RIGHT'],[predicted,{'grid':prefix['after']['grid']},{'status':'unscored'}]]
   assert [(x['method'],x['args_sha256']) for x in ev['rpc_methods']]==[(name,argsha(a)) for name,a in zip(['initialize','predict','assimilate'],expected_args)]
   tree=ast.parse(source);assert [x.name for x in tree.body if isinstance(x,ast.FunctionDef)]==['initialize','predict','assimilate']
 assert all(ledger['arms'][arm][k]==v for k,v in totals.items());assert root['usage']=={k:totals[k] for k in ['prompt_tokens','completion_tokens']};assert totals['prompt_tokens']<=600000 and totals['completion_tokens']<=24576
 assert root['inspections']==n-1 and root['submissions']==1 and root['repair']=={'accepted_after_repair':False,'failed_submissions':0}
 result['arms'][arm]={'totals':totals,'cost_usd':str(cost),'cached_prompt_tokens':cached,'calls':calls,'result':root,'supervision':sup,'work_item':progress['work_item'],'source_digest':f['source_digest'],'source_representation':'actual JSON array of lines, joined with newline plus trailing newline','evaluation':ev,'rpc_argument_digests_independently_rederived':True}
assert first['off']==first['on'];shared=sha((json.dumps(first['off'],sort_keys=True,indent=2,allow_nan=False)+'\n').encode());assert shared==ledger['off_initial_shared_sha256']==ledger['arms']['on']['initial_shared_sha256']
assert cumulative==D(ledger['known_cost_usd']) and cumulative<2
oldledger=js(OLD/'live/ledger.json');result['prior_pair']={'cost_usd':oldledger['known_cost_usd'],'arms':{a:js(OLD/f'live/{a}/RESULT.json') for a in ['off','on']}}
result.update(pair_cost_usd=str(cumulative),unknown_exposure_usd='0',initial_parity=True,initial_shared_sha256=shared)
assert all(sha(Path(p).read_bytes())==h for p,h in seen.items())
(O/'CHECKS.json').write_text(json.dumps(result,indent=2)+'\n');(O/'INPUT-HASHES.json').write_text(json.dumps(seen,indent=2)+'\n')
print(json.dumps({'inventory_sha256':ih,'pair_cost':str(cumulative),'prior_files_unchanged':len(snap['files']),'r10_sources_unchanged':len(m['sources']),'arms':{a:{k:r[k] for k in ['totals','cost_usd','source_digest']} for a,r in result['arms'].items()}},indent=2))
