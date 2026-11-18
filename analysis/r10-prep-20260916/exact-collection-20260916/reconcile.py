"""Offline descriptive reconciliation of exact numeric-version downloaded evidence."""
import ast,collections,hashlib,importlib.util,json,re,sys
from pathlib import Path
OUT=Path(__file__).resolve().parent;BASE=OUT.parent;ROOT=BASE.parents[1];RAW=OUT/'raw';WORK=RAW/'model-runtime/solver-workspaces';P=BASE/'r10a-package'
assert (OUT/'COLLECTION-REMAINING.json').is_file(), 'Collection still in progress; do not read in-flight raw files'
def js(p):return json.loads(p.read_text())
def rows(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()] if p.is_file() else []
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,d):
 with (OUT/name).open('w') as f:json.dump(d,f,indent=2);f.write('\n')
# Every present output must retain a matching successful exact request/response receipt.
receipts=[]
for rd in ['receipts','receipts-remaining']:
 for rp in sorted((OUT/rd).glob('*-response.json')):
  r=js(rp);q=js(rp.with_name(rp.name.replace('-response','-request')))
  assert q['fields']==dict(owner_slug='pranavagrawai',kernel_slug='r10a-cd82-constructor-repair',version_number=1,file_path=r['file_path'])
  assert type(q['fields']['version_number']) is int
  if r['status']=='RETRIEVED':assert sha(OUT/r['response_file'])==r['sha256'] and (OUT/r['response_file']).stat().st_size==r['bytes']
  receipts.append({**r,'receipt':str(rp.relative_to(OUT))})
assert len({r['file_path'] for r in receipts})==len(receipts),'No path retries permitted'
launcher=js(RAW/'model-runtime/official-launcher-result.json');assert sha(RAW/'model-runtime/official-launcher-result.json')==js(BASE/'version-provenance-review/session-mapping.json')['source_sha256']
checks={'retrieval':{'requests':len(receipts),'retrieved':sum(r['status']=='RETRIEVED' for r in receipts),'failed':[r for r in receipts if r['status']!='RETRIEVED'],'total_bytes':sum(r.get('bytes',0) for r in receipts),'all_present_hashes_matched':True,'no_duplicate_paths':True},'arms':{}}
progress=rows(WORK/'raw/game-progress.jsonl');lookup={g['arm']:g for g in progress}
for g in launcher['games']:
 arm=g['arm'];session=WORK/g['workspace']/g['sessions'][0]['session_dir'];ipc=session/'ipc';trans=rows(ipc/'transitions.jsonl');states=rows(ipc/'states.jsonl');actions=rows(ipc/'actions.jsonl');record_path=WORK/g['workspace']/'recordings'/g['official']['recording_files'][0];record_all=rows(record_path);record=[r['data'] for r in record_all if 'frame' in r.get('data',{})]
 details=[]
 for i,t in enumerate(trans):
  r=record[i];before_grid=[[int(c,16) for c in row] for row in states[i]['grid_rows']];after_grid=[[int(c,16) for c in row] for row in states[i+1]['grid_rows']]
  details.append({'step':t['step'],'action':t['action'],'before_matches_state':t['before']['grid']==before_grid,'after_matches_state':t['after']['grid']==after_grid,'after_matches_official_grid':t['after']['grid']==r['frame'][-1],'after_levels_match_official':t['after']['levels_completed']==r['levels_completed'],'after_state_matches_official':t['after']['state']==r['state'],'official_action_id':r.get('action_input',{}).get('id'),'before_chain_matches':i==0 or t['before']==trans[i-1]['after'],'changed_cells':sum(a!=b for ar,br in zip(t['before']['grid'],t['after']['grid']) for a,b in zip(ar,br))})
 preds=rows(ipc/'model_predictions.jsonl');feedback=rows(ipc/'model_feedback.jsonl');step_checks=rows(ipc/'step_checks.jsonl');lifecycle=rows(ipc/'request_lifecycle.jsonl');calls=rows(ipc/'model_calls.jsonl');budget=js(ipc/'budget_exhausted.json');host=rows(session/'host-events.jsonl')
 checks['arms'][arm]={'session':str(session.relative_to(RAW)),'transitions':len(trans),'state_rows':len(states),'official_frames':len(record),'official_empty_data_rows':len(record_all)-len(record),'official_recording_sha256':sha(record_path),'submitted_action_names':[a for r in actions for a in r['actions']],'transition_actions':[r['action'] for r in trans],'final_levels':record[-1]['levels_completed'],'final_state':record[-1]['state'],'launcher_progress_counts_match':len(trans)==g['actions']==lookup[arm]['actions']==len(record),'grid_and_progress_pairs':details,'all_grid_progress_pairs_match':all(all(d[k] for k in ['before_matches_state','after_matches_state','after_matches_official_grid','after_levels_match_official','after_state_matches_official','before_chain_matches']) for d in details),'official_action_labels_match_ipc':False if any(d['official_action_id']==0 and d['action']!='RESET' for d in details) else None,'official_action_limit':'All recording action_input.id fields are 0; action labels come from IPC. Official successor grids and levels independently match. No scorecard run provided.','budget_stop':budget,'stop':js(session/'stop.json'),'incumbent_model_calls':len(calls),'incumbent_roles':dict(collections.Counter(r['request_purpose'] for r in calls)),'reasoning_present_calls':sum(bool(r.get('reasoning_present')) for r in calls),'request_lifecycle_events':dict(collections.Counter(r['event'] for r in lifecycle)),'admissions_after_deadline':sum(r['event']=='admitted' and r['at_monotonic']>r['admission_deadline'] for r in lifecycle),'prediction_statuses':dict(collections.Counter(r['status'] for r in preds)),'feedback_statuses':dict(collections.Counter(r['status'] for r in feedback)),'step_check_statuses':dict(collections.Counter(r['status'] for r in step_checks)),'host_event_counts':dict(collections.Counter(r['event']+':'+r['status'] for r in host)),'frozen_prediction_count':sum(r['event']=='query_frozen_predict' and r['status']=='COMPLETED' for r in host),'frozen_assimilation_count':sum(r['event']=='assimilate_frozen' and r['status']=='COMPLETED' for r in host)}
# Constructor chain is inspected statically; never execute provider code locally.
bgame=launcher['games'][1];session=WORK/bgame['workspace']/bgame['sessions'][0]['session_dir'];events=rows(session/'constructor-job/tool-events.jsonl');te=[]
for i,e in enumerate(events,1):
 fn=e['call']['function'];args=json.loads(fn['arguments']);result=json.loads(e['rendered_result_message']['content']);item={'ordinal':i,'tool_call_id':e['call']['id'],'tool':fn['name'],'wall_seconds':e['wall_seconds'],'result':result}
 if fn['name']=='submit_candidate':
  source=args['candidate_source'];tree=ast.parse(source);digest=hashlib.sha256(source.encode()).hexdigest();item.update(source_json_type=type(source).__name__,source_sha256=digest,top_level_ast=[type(n).__name__ for n in tree.body],top_level_function_names=[n.name for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))],source_digest_matches_evaluation=digest==result['visible_evaluation']['model_sha256'])
 te.append(item)
t=rows(session/'ipc/transitions.jsonl')[0];diff=[(r,c,a,b) for r,(ar,br) in enumerate(zip(t['before']['grid'],t['after']['grid'])) for c,(a,b) in enumerate(zip(ar,br)) if a!=b];expected=f"step {t['step']} action {t['action']}\ndims before 64 64\ndims after 64 64\nnum diffs {len(diff)}\n"+''.join(str(d)+'\n' for d in diff[:80]);actual=json.loads(events[2]['rendered_result_message']['content'])['stdout']
construction=js(session/'CONSTRUCTION.json');checks['constructor']={'parent_construction':construction,'parent_child_result_equal':js(session/'RESULT.json')==js(session/'constructor-job/RESULT.json'),'root_child_tool_events_equal':sha(session/'tool-events.jsonl')==sha(session/'constructor-job/tool-events.jsonl'),'inspection_matches_exact_prefix_transition':expected==actual,'events':te,'construction_repair_rate':{'numerator':0,'denominator':1,'value':0.0},'failed_submission_count':3,'accepted_submission_count':0,'authoritative_freeze_count':0,'child_freeze_count':0,'frozen_prospective_queries':0,'frozen_assimilations':0,'frozen_unscored_commitments':0,'pending_frozen_predictions_at_stop':0,'prospective_binding':'UNEXERCISED_NO_FREEZE','host_event':rows(session/'host-events.jsonl')[0]}
# Accounting: independent sums, raw body hashes, wire-message hashes and purpose binding.
req=rows(WORK/'raw/requests.jsonl');resp=rows(WORK/'raw/responses.jsonl');latency=rows(WORK/'raw/latency.jsonl');assert [r['call'] for r in req]==[r['call'] for r in resp]==list(range(1,106))
missing_req=[];missing_resp=[];bad=[];usage=collections.defaultdict(lambda:collections.Counter());constructor_calls=[];wire_checks=[];byreceipt={r['call']:r for r in launcher['local_requests']['receipts']}
for q,r in zip(req,resp):
 i=q['call'];bodypath=WORK/f'raw/raw-http/{i:02d}.body';wirepath=WORK/f'raw/raw-requests/{i:03d}.wire.json';body=r['response'];u=body['usage'];usage[q['arm']+':'+q['purpose']].update({k:u[k] for k in ['prompt_tokens','completion_tokens']})
 if not bodypath.is_file():missing_resp.append(i)
 elif js(bodypath)!=body or sha(bodypath)!=byreceipt[i]['raw_sha256']:bad.append({'call':i,'check':'raw_response'})
 if not wirepath.is_file():missing_req.append(i);continue
 wire=js(wirepath);messages=json.dumps(wire.get('messages'),sort_keys=True).encode();slim=q['request'];row={'call':i,'model_matches':wire['model']==slim['model'],'wire_messages_sha256':hashlib.sha256(messages).hexdigest(),'declared_sha256':slim.get('messages_sha256'),'max_tokens':wire.get('max_tokens'),'purpose':q['purpose'],'thinking_disabled':(wire.get('chat_template_kwargs') or {}).get('enable_thinking') is False}
 row['messages_digest_matches']=row['wire_messages_sha256']==row['declared_sha256'];wire_checks.append(row)
 if q['purpose']=='constructor':
  tools=[m for m in wire['messages'] if m.get('role')=='tool'];message=body['choices'][0]['message'];calls=message.get('tool_calls') or [];constructor_calls.append({'call':i,'wire_request_sha256':sha(wirepath),'raw_response_sha256':sha(bodypath) if bodypath.is_file() else None,'usage':u,'max_tokens':wire.get('max_tokens'),'prior_tool_feedback_ids':[m.get('tool_call_id') for m in tools],'feedback_matches_exact_tool_events':all(any(m==e['rendered_result_message'] for e in events) for m in tools),'output_tool_call_ids':[c['id'] for c in calls],'output_matches_tool_events':all(any(c==e['call'] for e in events) for c in calls),'request_time':q['time'],'http_elapsed_seconds':next(l['elapsed_seconds'] for l in latency if l['call']==i)})
known={k:sum(v[k] for v in usage.values()) for k in ['prompt_tokens','completion_tokens']};checks['accounting']={'admitted':len(req),'responses':len(resp),'role_counts':dict(collections.Counter(q['arm']+':'+q['purpose'] for q in req)),'usage_by_arm_role':dict(usage),'known_usage':known,'usage_matches_launcher':known==launcher['local_requests']['known_usage'],'missing_raw_request_calls':missing_req,'missing_raw_response_calls':missing_resp,'raw_response_mismatches':bad,'wire_checks':wire_checks,'constructor_calls':constructor_calls,'all_completion_caps_respected':all(r['response']['usage']['completion_tokens']<=q['request']['max_tokens'] for q,r in zip(req,resp)),'latency_max_by_role':{role:max(x['elapsed_seconds'] for x in latency if x['purpose']==role) for role in {x['purpose'] for x in latency}}}
# Recorded identity hashes compared to frozen expected bytes, not merely status labels.
mv=js(RAW/'model-runtime/model-verification.json');expected={r['path']:r for r in js(P/'model-metadata.json')['files']};model_bad=[]
for row in mv['files']:
 e=expected[row['path']]
 if row['expected_sha256']!=e['sha256'] or row['expected_size']!=e['size']:model_bad.append(row['path'])
 if row['status']=='matched' and (row['actual_sha256']!=e['sha256'] or row['actual_size']!=e['size']):model_bad.append(row['path'])
diff=js(WORK/'raw/ARM-DIFF.json');shared_bad=[n for n,h in diff['shared_agent_sha256'].items() if sha(P/n)!=h]
checks['identity']={'runtime_model_id':mv['model_id'],'runtime_model_revision':mv['revision'],'model_path':mv['model_path'],'model_file_statuses':dict(collections.Counter(r['status'] for r in mv['files'])),'model_expected_actual_hash_mismatches':model_bad,'structure':mv['structure'],'game':js(WORK/'raw/GAME-PREFLIGHT.json'),'gpu':js(WORK/'raw/GPU.json')['device'],'seal':js(WORK/'raw/SEAL.json'),'framework':js(WORK/'raw/FRAMEWORK.json'),'shared_agent_hash_mismatches':shared_bad,'arm_difference_receipt':diff,'bootstrap_game_preflight':js(RAW/'actor-critic-construction/GAME-PREFLIGHT.json'),'limit':'Downloaded runtime testimony with internally reconciled hashes; no independent reread of mounted model weights or physical GPU.'}
tree=ast.parse((P/'gameplay_play.py').read_text());constants={}
for node in tree.body:
 if isinstance(node,ast.Assign):
  for target in node.targets:
   if isinstance(target,ast.Name) and target.id in ['OFFICIAL_HASHES','OFFICIAL_REVISION']:
    constants[target.id]=ast.literal_eval(node.value)
framework=checks['identity']['framework'];checks['identity']['framework_hashes_match_frozen_contract']=framework['official']==constants['OFFICIAL_HASHES'] and framework['revision']==constants['OFFICIAL_REVISION']
checks['identity']['worker_entry_hashes_match_frozen']={a:sha(P/('nooa-join/headless.py' if a=='A' else 'headless.py'))==h for a,h in diff['worker_entry_sha256'].items()}
checks['teardown']={'shutdown':js(RAW/'model-runtime/shutdown.json'),'bootstrap_reap':js(RAW/'actor-critic-construction/SHUTDOWN-REAP.json'),'process_guard':js(RAW/'actor-critic-construction/PROCESS-GUARD.json'),'system_after_gpu':js(RAW/'model-runtime/system-after.json')['gpu']}
for name in ['lifecycle-owned-before.json','lifecycle-owned-after.json','lifecycle-observer.json','teardown-log-classification.json','listeners-after.json','gpu-processes-after.json']:
 p=RAW/'model-runtime'/name
 if p.is_file():checks['teardown'][name]=js(p)
p=RAW/'model-runtime/lifecycle-polls.jsonl'
if p.is_file():
 polls=rows(p);closed=[r for r in polls if r['closed']];checks['teardown']['poll_summary']={'polls':len(polls),'closed':len(closed),'first_closed_elapsed':closed[0]['elapsed_since_term_s'] if closed else None,'quiet_span':closed[-1]['monotonic_s']-closed[0]['monotonic_s'] if closed else None,'closed_remain_closed':all(r['closed'] for r in polls[polls.index(closed[0]):]) if closed else False}
p=RAW/'model-runtime/server.log'
if p.is_file():
 log=p.read_text();checks['accounting']['server_accesses']={'chat':len(re.findall(r'"POST /v1/chat/completions HTTP/1\.1"',log)),'chat_200':len(re.findall(r'"POST /v1/chat/completions HTTP/1\.1" 200',log)),'freeze_gc_200':len(re.findall(r'"POST /freeze_gc HTTP/1\.1" 200',log))}
protected=js(OUT/'PROTECTED-INPUT-HASHES.json');checks['protected_input_drift']=[n for n,h in protected.items() if sha(ROOT/n)!=h];assert not checks['protected_input_drift'];assert not (BASE/'r10a-cd82-20260916/launch.json').exists()
save('RECONCILIATION.json',checks)
print(json.dumps({'retrieved':checks['retrieval']['retrieved'],'requests':checks['retrieval']['requests'],'progress':{a:{k:v for k,v in d.items() if k in ['transitions','final_levels','final_state','all_grid_progress_pairs_match']} for a,d in checks['arms'].items()},'constructor_status':construction['construction_status'],'raw_requests_missing':missing_req,'raw_responses_missing':missing_resp,'known_usage':known,'wire_hash_match_count':sum(r['messages_digest_matches'] for r in wire_checks),'protected_drift':checks['protected_input_drift']},indent=2))
