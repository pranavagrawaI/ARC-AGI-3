"""Offline independent result audit. Never execute candidate code or write inputs."""
import ast, collections, hashlib, json, runpy
from pathlib import Path
OUT=Path(__file__).resolve().parent
BASE=OUT.parent; C=BASE/'exact-collection-20260916'; RAW=C/'raw'; W=RAW/'model-runtime/solver-workspaces'; P=BASE/'r10a-package'
seen={}
def read(p):
 p=Path(p); b=p.read_bytes(); seen[str(p)]=hashlib.sha256(b).hexdigest(); return b
def js(p):return json.loads(read(p))
def rows(p):return [json.loads(l) for l in read(p).splitlines() if l.strip()]
def digest(b):return hashlib.sha256(b).hexdigest()
read(C/'RESULT.md'); js(C/'RECONCILIATION.json')
receipts=[]
for folder in ['receipts','receipts-remaining']:
 for p in sorted((C/folder).glob('*-response.json')):
  r=js(p); q=js(p.with_name(p.name.replace('-response','-request')))
  assert q['fields']==dict(owner_slug='pranavagrawai',kernel_slug='r10a-cd82-constructor-repair',version_number=1,file_path=r['file_path'])
  if r['status']=='RETRIEVED':
   b=read(C/r['response_file']); assert digest(b)==r['sha256'] and len(b)==r['bytes']
  receipts.append(r)
assert len(set(r['file_path'] for r in receipts))==len(receipts)==330
launcher=js(RAW/'model-runtime/official-launcher-result.json'); result={'retrieval':dict(attempts=330,retrieved=sum(r['status']=='RETRIEVED' for r in receipts)),'arms':{}}
for g in launcher['games']:
 s=W/g['workspace']/g['sessions'][0]['session_dir']; ts=rows(s/'ipc/transitions.jsonl'); states=rows(s/'ipc/states.jsonl'); rec=rows(W/g['workspace']/'recordings'/g['official']['recording_files'][0]); frames=[r['data'] for r in rec if 'frame' in r.get('data',{})]
 assert len(ts)==len(frames)==g['actions']; assert rec[-1]['data']=={}
 for i,(t,f) in enumerate(zip(ts,frames)):
  assert t['before']['grid']==[[int(c,16) for c in r] for r in states[i]['grid_rows']]
  assert t['after']['grid']==[[int(c,16) for c in r] for r in states[i+1]['grid_rows']]==f['frame'][-1]
  assert t['after']['levels_completed']==f['levels_completed']==0
  assert t['after']['state']==f['state']
 h=rows(s/'host-events.jsonl') if (s/'host-events.jsonl').exists() else [];
 assert h or g['arm']=='A'; assert not any(x['event'] in ['query_frozen_predict','assimilate_frozen'] and x['status']=='COMPLETED' for x in h)
 pred=rows(s/'ipc/model_predictions.jsonl');feed=rows(s/'ipc/model_feedback.jsonl')
 assert all(x['status']=='unavailable' for x in pred+feed)
 result['arms'][g['arm']]={'transitions':len(ts),'final_levels':frames[-1]['levels_completed'],'final_state':frames[-1]['state'],'official_action_ids':sorted(set(f['action_input']['id'] for f in frames)),'prediction_count':len(pred),'feedback_count':len(feed),'frozen_completed_events':0}
s=W/'arm-B/cd82-fb555c5d-b7azqbdf'; events=rows(s/'tool-events.jsonl');assert read(s/'tool-events.jsonl')==read(s/'constructor-job/tool-events.jsonl')
r=js(s/'RESULT.json');assert r==js(s/'constructor-job/RESULT.json');assert (r['construction_status'],r['frozen'],r['inspections'],r['submissions'],r['model_requests'])==('BUDGET_EXHAUSTED',False,3,3,6)
assert r['prospective']=='not_run' and r['supported_model'] is False
req=rows(W/'raw/requests.jsonl');resp=rows(W/'raw/responses.jsonl');bycall={r['call']:r for r in launcher['local_requests']['receipts']}
assert [q['call'] for q in req]==[r['call'] for r in resp]==list(range(1,106))
assert [q['call'] for q in req if q['purpose']=='constructor']==list(range(74,80))
usage=collections.Counter(); missing=[]; constructor=[]
for q,rsp in zip(req,resp):
 n=q['call']; b=read(W/f'raw/raw-http/{n:02}.body'); body=json.loads(b);assert body==rsp['response']; assert digest(b)==bycall[n]['raw_sha256']
 for k in ['prompt_tokens','completion_tokens']:usage[k]+=body['usage'][k]
 assert body['usage']['completion_tokens']<=q['request']['max_tokens']
 path=W/f'raw/raw-requests/{n:03}.wire.json'
 if not path.exists():missing.append(n);continue
 wire=js(path);assert digest(json.dumps(wire['messages'],sort_keys=True).encode())==q['request']['messages_sha256']
 if 74<=n<=79:
  i=n-74;e=events[i];assert body['choices'][0]['message']['tool_calls']==[e['call']]
  prior=[m for m in wire['messages'] if m['role']=='tool'];assert prior==[v['rendered_result_message'] for v in events[:i]]
  tools={t['function']['name']:t['function'] for t in wire['tools']}
  name=e['call']['function']['name'];assert name in tools
  entry={'call':n,'tool':name,'exact_prior_feedback':True,'outbound_tools':list(tools)}
  if name=='submit_candidate':
   schema=tools[name]['parameters']['properties']['candidate_source'];assert schema['oneOf']==[{'type':'string'},{'type':'array','items':{'type':'string'}}]
   source=json.loads(e['call']['function']['arguments'])['candidate_source'];assert isinstance(source,str)
   tree=ast.parse(source);assert len(tree.body)==1 and isinstance(tree.body[0],ast.Expr) and isinstance(tree.body[0].value,ast.List)
   assert all(isinstance(x,ast.Constant) and isinstance(x.value,str) for x in tree.body[0].value.elts)
   evaluation=json.loads(e['rendered_result_message']['content'])['visible_evaluation'];assert digest(source.encode())==evaluation['model_sha256'];assert "NameError: name 'initialize' is not defined" in evaluation['visible']['error']
   assert evaluation['visible']['status']=='failed'
   entry.update(source_sha256=digest(source.encode()),source_type='string',ast='Expr(List(Constant strings))',function_definitions=0,source_preserved=True)
  else: assert json.loads(e['rendered_result_message']['content'])['returncode']==0
  constructor.append(entry)
assert missing==[24];assert dict(usage)==launcher['local_requests']['known_usage']
t=rows(s/'ipc/transitions.jsonl')[0];diff=[(i,j,a,b) for i,(ar,br) in enumerate(zip(t['before']['grid'],t['after']['grid'])) for j,(a,b) in enumerate(zip(ar,br)) if a!=b]
text=f"step {t['step']} action {t['action']}\ndims before 64 64\ndims after 64 64\nnum diffs {len(diff)}\n"+''.join(str(d)+'\n' for d in diff[:80]);assert text==json.loads(events[2]['rendered_result_message']['content'])['stdout'];assert len(diff)==201
read(P/'local_wire.py');project=runpy.run_path(str(P/'local_wire.py'))['project_messages'];m=rows(W/'arm-A/cd82-fb555c5d-rmrz0g5o/ipc/model_calls.jsonl')[23]['request_messages']; projected=digest(json.dumps(project(m),sort_keys=True).encode());assert projected==req[23]['request']['messages_sha256'];assert req[23]['purpose']=='actor' and req[23]['arm']=='A'
latency=rows(W/'raw/latency.jsonl'); ctor_latency=[x['elapsed_seconds'] for x in latency if x['call'] in range(74,80)]; assert max(ctor_latency)<30
ctor_usage={k:sum(resp[i-1]['response']['usage'][k] for i in range(74,80)) for k in ['prompt_tokens','completion_tokens']};assert ctor_usage==js(s/'RESULT.json')['usage'];assert ctor_usage['completion_tokens']<24576
construction=js(s/'CONSTRUCTION.json');result['constructor_accounting']={'usage':ctor_usage,'max_http_seconds':max(ctor_latency),'construction':construction}
result.update(constructor=constructor,constructor_result=js(s/'RESULT.json'),inspection_changed_cells=201,accounting={'calls':105,'usage':dict(usage),'missing_raw_requests':missing,'missing024_arm_role':'A actor','derived024_message_digest':projected})
before=js(RAW/'model-runtime/lifecycle-owned-before.json');after=js(RAW/'model-runtime/lifecycle-owned-after.json');polls=rows(RAW/'model-runtime/lifecycle-polls.jsonl');closed=[r for r in polls if r['closed']];assert len(before['processes'])==5 and len(before['gpu_processes'])==2
assert not after['processes'] and not after['gpu_processes'] and not after['listeners']['inodes'] and not after['listeners']['unresolved_inodes'];assert all(r['closed'] for r in polls[polls.index(closed[0]):]);assert len(closed)==7
log=read(RAW/'model-runtime/server.log');assert log.count(b'"POST /v1/chat/completions HTTP/1.1" 200')==106
result['teardown']={'processes_before':5,'processes_after':0,'gpu_processes_before':2,'gpu_processes_after':0,'closed_scans':7,'first_closed_seconds':closed[0]['elapsed_since_term_s'],'quiet_span_seconds':closed[-1]['monotonic_s']-closed[0]['monotonic_s'],'shutdown':js(RAW/'model-runtime/shutdown.json'),'bootstrap':js(RAW/'actor-critic-construction/SHUTDOWN-REAP.json'),'guard':js(RAW/'actor-critic-construction/PROCESS-GUARD.json')}
for n in ['constructor.py','integration_boundary.py']:read(P/n)
for p in [BASE/'independent-launch-review/REPORT.md',BASE/'independent-launch-review/INCIDENT-ADDENDUM-20260916.md',BASE/'r10a-cd82-20260916/CONSUMED.json',BASE/'r10a-cd82-20260916/launch-review.json']:read(p)
assert all(digest(Path(p).read_bytes())==h for p,h in seen.items())
(OUT/'CHECKS.json').write_text(json.dumps(result,indent=2)+'\n');(OUT/'INPUT-HASHES.json').write_text(json.dumps(seen,indent=2)+'\n')
print(json.dumps({'status':'PASS_BOUNDED_FACTUAL_CHECKS','arms':result['arms'],'calls':105,'constructor_calls':[r['call'] for r in constructor],'input_files_bound':len(seen)},indent=2))
