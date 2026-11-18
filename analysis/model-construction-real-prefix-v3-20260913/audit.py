#!/usr/bin/env python3
"""Independent durable-artifact audit; never invokes the model or evaluator."""
import hashlib, json
from decimal import Decimal
from accounting import receipts
from pathlib import Path
HERE=Path(__file__).resolve().parent
read=lambda n:json.loads((HERE/n).read_text())
def lines(n): return [json.loads(x) for x in (HERE/n).read_text().splitlines()] if (HERE/n).exists() else []
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
r=read('RESULT.json'); started=read('STARTED.json')
req=[json.loads(p.read_text()) for p in sorted((HERE/'raw/requests').glob('*.json'))]
res=[json.loads(p.read_text()) for p in sorted((HERE/'raw/responses').glob('*.json'))]
events=lines('raw/tool_events.jsonl'); attempts=lines('raw/request_attempts.jsonl'); calls=lines('raw/model_calls.jsonl')
trans=read('raw/transcript.json') if (HERE/'raw/transcript.json').exists() else []
budgets=[json.loads(x['messages'][-1]['content'])['budget'] for x in req]
account=receipts(res,len(req))
usage_valid=account['valid_receipts']
prompt=account['prompt_tokens']; completion=account['completion_tokens']; cost=account['cost']
failed=[x for x in attempts if x['status']=='failed']
terminal=[x for x in attempts if x['status'] in ('completed','failed')]
selection=read('PROVIDER-SELECTION.json')
interrupted=bool(failed) and r['classification']=='INTEGRATION_INVALID'

check={
'prelaunch_passed':read('PRELAUNCH.json')['passed'] is True,
'frozen_sources_unchanged':all(digest(HERE/n)==h for n,h in started['artifact_sha256'].items()),
'baseline_unchanged':all(digest(HERE.parent/'model-construction-real-prefix-20260913'/n)==h for n,h in read('BASELINE-SHA256.json').items()),
'bounded_request_attempts':len(req)<=16 and [x['index'] for x in attempts if x['status']=='started']==list(range(len(req))),
'no_retries':len(attempts)==len(req)+len([x for x in attempts if x['status'] in ['completed','failed']]) and len({x['index'] for x in attempts if x['status']=='started'})==len(req),
'completed_counts':len(res)==len(calls)==len([x for x in attempts if x['status']=='completed']) and len(req)==r.get('model_requests'),
'failure_stops_episode':not failed or (len(failed)==1 and failed[0]['index']==len(req)-1 and interrupted),
'latency_all_attempts':len(terminal)==len(req) and all(type(x.get('wall_seconds')) in (int,float) and 0<=x['wall_seconds']<900 for x in terminal),
'v2_unchanged':all(digest(HERE.parent/'model-construction-real-prefix-v2-20260913'/n)==h for n,h in read('V2-SHA256.json').items()),
'actual_provider_matches_pin':all(x.get('provider')==selection['selected']['provider_name'] for x in res),
'usage_valid_or_explicit_unknown_abort':usage_valid or (interrupted and not res and prompt is None and completion is None and cost is None),
'actual_prompt_cap_or_unknown_abort':(prompt is not None and prompt<=300000) or (interrupted and prompt is None),
'actual_completion_cap_or_unknown_abort':(completion is not None and completion<=16384) or (interrupted and completion is None),
'usage_reconciles':r.get('usage')==({'prompt_tokens':prompt,'completion_tokens':completion} if account['complete'] else None),
'request_allowance_reconciles':sum(x['max_tokens'] for x in req)==r.get('requested_tokens') if not interrupted else all(q['max_tokens']==a['requested_tokens'] for q,a in zip(req,[x for x in attempts if x['status']=='started'])),
'model_unchanged':all(x['model']=='qwen/qwen3.8-27b' for x in req+res),
'reasoning_off':all(x['reasoning']=={'enabled':False} for x in req) and all((x.get('usage',{}).get('completion_tokens_details') or {}).get('reasoning_tokens',0)==0 for x in res) and not r.get('reasoning_used',False),
'provider_cost_ceiling':all(x['provider']==selection['routing'] for x in req) and (cost is None or cost<=Decimal('.50')),
'request_wire_bodies_match':all(json.loads((HERE/'raw/requests'/f'{i:02d}.body').read_bytes())==q for i,q in enumerate(req)),
'raw_bodies_match':all(json.loads((HERE/'raw/responses'/f'{i:02d}.body').read_bytes())==x for i,x in enumerate(res)),
'artifact_hashes_match':all(digest(HERE/c[k+'_path'])==c[k+'_sha256'] for c in calls for k in ['request','response']),
'exact_transcript':trans==[{'request':a,'response':b} for a,b in zip(req,res)] or (interrupted and not (HERE/'raw/transcript.json').exists()),
'exact_tool_calls':all(e['call'] in [c for x in res for c in x['choices'][0]['message'].get('tool_calls',[])] for e in events),
'exact_rendered_tool_results':all(m in [e['rendered_result_message'] for e in events] for x in req for m in x['messages'] if m['role']=='tool'),
'no_suffix_hash_in_requests':all(started['sealed_suffix_sha256'] not in json.dumps(x) for x in req),
'wall_cap':r.get('elapsed_seconds',1000)<900,
'inspection_and_submission_caps':sum(e['call']['function']['name']=='execute_python' for e in events)<=10 and sum(e['call']['function']['name']=='submit_candidate' for e in events)<=4,
'commit_disallows_inspection':all(b['phase']!='commit' or 'execute_python' not in [t['function']['name'] for t in q['tools']] for q,b in zip(req,budgets)),
'completion_reserved_before_each_request':all(q['max_tokens']<=b['completion_tokens_remaining'] for q,b in zip(req,budgets)),
'precall_prompt_and_spend_reservations':all(len(json.dumps(q,ensure_ascii=False).encode())+4096<=b['prompt_tokens_remaining'] and (len(json.dumps(q,ensure_ascii=False).encode())+4096)*1e-6+q['max_tokens']*1e-5<=b['dollars_remaining_conservative']+1e-6 for q,b in zip(req,budgets)),
}
for i,b in enumerate(budgets):
    check[f'telemetry_{i}_actual_charge']=b['requests_remaining']==16-i and b['completion_tokens_remaining']==16384-sum(x['usage']['completion_tokens'] for x in res[:i]) and b['prompt_tokens_remaining']==300000-sum(x['usage']['prompt_tokens'] for x in res[:i])
commits=[b for b in budgets if b['phase']=='commit']
check['commit_reserve_intact']=not commits or (commits[0]['requests_remaining']>=2 and commits[0]['completion_tokens_remaining']>=4096)
if (HERE/'raw/FROZEN.json').exists():
    f=read('raw/FROZEN.json'); source=(HERE/'raw/frozen_candidate.py').read_text()
    check['frozen_source_is_model_submission']=any(e['call']['function']['name']=='submit_candidate' and json.loads(e['call']['function']['arguments']).get('candidate_source')==source for e in events)
    check['frozen_identity_and_fresh_belief']=digest(HERE/'raw/frozen_candidate.py')==f['model_sha256']==f['visible_evaluation']['model_sha256']==f['visible_evaluation']['belief_model_sha256']
    check['one_postfreeze_suffix_evaluation']=lines('raw/suffix_attempts.jsonl')==[{'status':s,'model_sha256':f['model_sha256']} for s in ['started','completed']]
    check['suffix_result_matches']=r['sealed_suffix_evaluation']==read('raw/sealed_suffix_evaluation.json')
    check['no_request_after_freeze']=all((HERE/'raw/requests'/f'{i:02d}.json').stat().st_mtime_ns <= (HERE/'raw/FROZEN.json').stat().st_mtime_ns for i in range(len(req)))
else:
    check['no_unfrozen_suffix']=r.get('sealed_suffix_evaluation') is None and not lines('raw/suffix_attempts.jsonl')
a={'passed':all(check.values()),'checks':check,'provider_reported_cost_usd':str(cost) if cost is not None else None,
   'prompt_tokens':prompt,'completion_tokens':completion,'request_count':len(req),
   'completed_requests':len(res), 'receipt_status':'complete' if account['complete'] else 'unknown_total_no_complete_receipts',
   'known_successful_usage':account['known_successful_usage'],
   'known_successful_cost_usd':str(account['known_successful_cost']) if account['known_successful_cost'] is not None else None,
   'attempt_latencies':terminal, 'selected_provider':selection['selected']['provider_name'],
   'actual_response_providers':sorted(set(x.get('provider','UNKNOWN') for x in res)),
   'per_request':[{'index':i,'latency_seconds':c['wall_seconds'],'prompt_tokens':res[i]['usage']['prompt_tokens'],'completion_tokens':res[i]['usage']['completion_tokens'],'cost_usd':res[i]['usage'].get('cost')} for i,c in enumerate(calls)],
   'scope':'Artifact consistency, not proof against every sandbox exploit or general competence.'}
(HERE/'AUDIT.json').write_text(json.dumps(a,indent=2)+'\n'); print(json.dumps(a)); raise SystemExit(0 if a['passed'] else 1)
