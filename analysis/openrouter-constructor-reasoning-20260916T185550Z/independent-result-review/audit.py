"""Offline evidence audit only; no candidate imports or execution."""
import ast,copy,hashlib,json
from decimal import Decimal as D
from pathlib import Path
O=Path(__file__).resolve().parent;B=O.parent;L=B/'live';inputs={}
def raw(p):
 b=p.read_bytes();inputs[str(p)]=hashlib.sha256(b).hexdigest();return b
def js(p):return json.loads(raw(p))
def sha(b):return hashlib.sha256(b).hexdigest()
def rows(p):return [json.loads(x) for x in raw(p).splitlines() if x.strip()]
inv=js(B/'INVENTORY.json');ih=sha(raw(B/'INVENTORY.json'));assert ih=='fa134897d4c592335ef9f7f047838773bf0c5cb4e74ca24921ee6d402c215dee'
for name,h in inv['files'].items():assert sha(raw(B/name))==h
ledger=js(L/'ledger.json');pair=js(L/'RESULT.json');assert pair['ledger']==ledger and pair['inventory_sha256']==ih
assert not ledger['stopped'] and D(ledger['unknown_exposure_usd'])==0
assert [(x['arm'],x['ordinal']) for x in ledger['attempts']]==[(a,i) for a in ['off','on'] for i in range(1,7)]
checks={'inventory_sha256':ih,'arms':{}};total=D(0);first={};prior_actual=D(0)
for arm in ['off','on']:
 C=L/arm/'child';events=rows(C/'tool-events.jsonl');progress=js(C/'PROGRESS.json');root=js(L/arm/'RESULT.json');child=js(C/'RESULT.json');sup=js(L/arm/'SUPERVISION.json')
 assert {k:v for k,v in root.items() if k!='authority'}==child
 assert root['construction_status']=='BUDGET_EXHAUSTED' and not root['frozen'] and root['prospective']=='not_run'
 assert progress['evaluation'] is None and progress['candidate_source'] is None
 assert not sup['hard_timeout'] and sup['returncode']==0 and sup['reaped'] and sup['elapsed_seconds']<245
 assert root['elapsed_seconds']<240
 assert len(events)==len(progress['transcript'])==6 and events==progress['tool_events']
 totals=dict(requests=6,prompt_tokens=0,completion_tokens=0,reasoning_tokens=0);cost=D(0);details=[];feedback=[];assistants=[];cached=0
 for i in range(1,7):
  qraw=raw(C/f'raw/{i:03}.request.json');rr=raw(C/f'raw/{i:03}.response.raw');q=json.loads(qraw);r=json.loads(rr);receipt=js(C/f'raw/{i:03}.receipt.json');record=next(x for x in ledger['attempts'] if x['arm']==arm and x['ordinal']==i)
  assert receipt==record and sha(qraw)==record['request_sha256'] and sha(rr)==record['response_sha256']
  assert receipt['http_status']==200 and receipt['error_type'] is None
  assert progress['transcript'][i-1]['response']==r
  logical=copy.deepcopy(progress['transcript'][i-1]['request']);logical.update(provider=q['provider'],reasoning=q['reasoning'],stream=False);assert logical==q
  assert q['provider']['only']==q['provider']['order']==['reka/fp8'] and q['provider']['allow_fallbacks'] is False and q['provider']['require_parameters'] is True
  assert q['reasoning']==({'enabled':False,'exclude':False} if arm=='off' else {'enabled':True,'exclude':False,'effort':'medium'})
  assert q['max_tokens']==4096
  if i==1:first[arm]=copy.deepcopy(q);first[arm].pop('reasoning')
  u=r['usage'];assert u==record['usage'];rt=u['completion_tokens_details']['reasoning_tokens'];assert u['total_tokens']==u['prompt_tokens']+u['completion_tokens'];assert 0<=rt<=u['completion_tokens']<=4096
  for k in ['prompt_tokens','completion_tokens']:totals[k]+=u[k]
  totals['reasoning_tokens']+=rt;cost+=D(str(u['cost']));cached+=u['prompt_tokens_details']['cached_tokens']
  calc=D(u['prompt_tokens']-u['prompt_tokens_details']['cached_tokens'])*D('.000000214')+D(u['prompt_tokens_details']['cached_tokens'])*D('.00000015')+D(u['completion_tokens'])*D('.00000255');assert calc==D(str(u['cost']))
  assert D(record['actual_cost_usd'])==D(str(u['cost']))
  exposure=D(len(qraw)+4096)*D('.000000214')+D(131072)*D('.00000255');assert exposure==D(record['exposure_usd']);assert prior_actual+exposure<=2;prior_actual+=D(str(u['cost']))
  assert u['prompt_tokens']<=record['reserved_prompt_tokens']
  m=r['choices'][0]['message'];assert r['choices'][0]['finish_reason']=='tool_calls';assert r['provider']=='Reka'
  meta=r['openrouter_metadata'];assert meta['attempt']==1 and meta['strategy']=='direct' and not meta.get('pipeline');assert [e for e in meta['endpoints']['available'] if e.get('selected')]==[{'provider':'Reka','model':'qwen/qwen3.8-27b-20260814','selected':True}]
  assert bool(m.get('reasoning') or m.get('reasoning_details'))==(arm=='on');assert (rt>0)==(arm=='on')
  assert [x for x in q['messages'] if x['role']=='tool']==feedback
  assert [x for x in q['messages'] if x['role']=='assistant']==assistants
  projected={'role':'assistant','content':m.get('content'),'tool_calls':[{'id':x['id'],'type':x['type'],'function':{k:x['function'][k] for k in ['name','arguments']}} for x in m['tool_calls']]}
  if m.get('reasoning_details'):projected['reasoning_details']=m['reasoning_details']
  elif m.get('reasoning'):projected['reasoning']=m['reasoning']
  assistants.append(projected);assert m['tool_calls']==[events[i-1]['call']];e=events[i-1];feedback.append(e['rendered_result_message']);args=json.loads(m['tool_calls'][0]['function']['arguments']);name=e['call']['function']['name'];result=json.loads(e['rendered_result_message']['content'])
  row={'call':i,'tool':name,'reasoning_tokens':rt,'completion_tokens':u['completion_tokens'],'cost':str(u['cost']),'feedback':result}
  if name=='submit_candidate':
   assert args['work_item']=='' and result=={'error':'ValueError: work_item must contain the exact required fields','status':'rejected'}
   tool=next(x['function'] for x in q['tools'] if x['function']['name']==name);assert tool['parameters']['properties']['work_item']=={'type':'object'}
   row.update(work_item_type='str',work_item_value='',argument_keys=list(args),candidate_source_type=type(args['candidate_source']).__name__)
   try:t=ast.parse(args['candidate_source']);row['static_source_ast']=[type(n).__name__ for n in t.body];row['static_functions']=[n.name for n in t.body if isinstance(n,ast.FunctionDef)]
   except SyntaxError as ex:row['static_source_error']=ex.msg
   # Verify the required nested packet was delivered, not merely a schema stub.
   packet=json.loads(q['messages'][1]['content']);assert isinstance(packet['required_submission']['work_item'],dict) and len(packet['required_submission']['work_item'])==9
  else:assert result['status']=='completed' and result['returncode']==0
  details.append(row)
 assert all(totals[k]==ledger['arms'][arm][k] for k in totals)
 assert root['usage']=={k:totals[k] for k in ['prompt_tokens','completion_tokens']}
 assert totals['prompt_tokens']<=600000 and totals['completion_tokens']<=24576
 assert not (L/arm/'FROZEN.json').exists() and not (C/'FROZEN.json').exists()
 checks['arms'][arm]={'totals':totals,'cost_usd':str(cost),'cached_prompt_tokens':cached,'result':root,'supervision':sup,'calls':details};total+=cost
assert first['off']==first['on'];sh=sha((json.dumps(first['off'],sort_keys=True,indent=2,allow_nan=False)+'\n').encode());assert sh==ledger['off_initial_shared_sha256']==ledger['arms']['on']['initial_shared_sha256']
assert total==D(ledger['known_cost_usd']) and total<2
checks.update(pair_cost_usd=str(total),initial_shared_sha256=sh,initial_parity=True,unknown_exposure_usd='0',no_root_or_child_freezes=True)
assert all(sha(Path(p).read_bytes())==v for p,v in inputs.items())
(O/'CHECKS.json').write_text(json.dumps(checks,indent=2)+'\n');(O/'INPUT-HASHES.json').write_text(json.dumps(inputs,indent=2)+'\n')
print(json.dumps({k:v for k,v in checks.items() if k!='arms'},indent=2));print(json.dumps({a:{k:d[k] for k in ['totals','cost_usd','cached_prompt_tokens']} for a,d in checks['arms'].items()},indent=2))
