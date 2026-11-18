#!/usr/bin/env python3
"""Post-run failure-specific audit, added after HTTP 429; no external calls.
Frozen audit.py is retained unchanged, including its failed completed-run assumptions.
"""
import hashlib, json
from pathlib import Path
HERE=Path(__file__).resolve().parent
read=lambda n:json.loads((HERE/n).read_text())
started=read('STARTED.json'); result=read('RESULT.json'); request=read('raw/requests/00.json')
error=read('raw/responses/00.error.body')
attempts=[json.loads(s) for s in (HERE/'raw/request_attempts.jsonl').read_text().splitlines()]
files=[str(p.relative_to(HERE/'raw')) for p in (HERE/'raw').rglob('*') if p.is_file()]
checks={
 'prelaunch_passed':read('PRELAUNCH.json')['passed'] is True,
 'source_card_tests_performance_unchanged':all(hashlib.sha256((HERE/n).read_bytes()).hexdigest()==h for n,h in started['artifact_sha256'].items()),
 'old_experiment_unchanged':all(hashlib.sha256((HERE.parent/'model-construction-real-prefix-20260913'/n).read_bytes()).hexdigest()==h for n,h in read('BASELINE-SHA256.json').items()),
 'one_started_one_failed_no_retry':len(attempts)==2 and [x['status'] for x in attempts]==['started','failed'] and [x['index'] for x in attempts]==[0,0],
 'exact_429_blocker':attempts[1]['http_status']==error['error']['code']==429 and error['error']['metadata']['limit_source']=='upstream_provider_shared_pool',
 'single_request_snapshot':sorted(p.name for p in (HERE/'raw/requests').glob('*.json'))==['00.json'],
 'no_successful_response_or_tools':not list((HERE/'raw/responses').glob('*.json')) and not (HERE/'raw/tool_events.jsonl').exists(),
 'no_frozen_candidate_or_suffix':not any((HERE/'raw'/n).exists() for n in ['FROZEN.json','frozen_candidate.py','sealed_suffix_evaluation.json','suffix_attempts.jsonl']),
 'invalid_classification':result['classification']=='INTEGRATION_INVALID' and result['model_requests']==1,
 'model_and_reasoning_policy':request['model']=='qwen/qwen3.8-27b' and request['reasoning']=={'enabled':False},
 'routing_no_retry_cost_ceiling':request['provider']=={'allow_fallbacks':False,'max_price':{'prompt':1,'completion':10}},
 'completion_reservation':request['max_tokens']==attempts[0]['requested_tokens']==4096,
 'prompt_reservation_and_dollar_bound':len(json.dumps(request,ensure_ascii=False).encode())+4096<=300000 and (len(json.dumps(request,ensure_ascii=False).encode())+4096)*1e-6+4096*1e-5<=.5,
 'budget_telemetry':json.loads(request['messages'][-1]['content'])['budget']['requests_remaining']==16,
 'wall_bounded':0<result['elapsed_seconds']<900,
 'unknown_usage_not_claimed_zero':'usage' not in error and 'cost' not in error,
}
audit={'passed':all(checks.values()),'checks':checks,'classification':'INTEGRATION_INVALID_UPSTREAM_RATE_LIMIT',
       'request_attempts':1,'completed_requests':0,'inspections':0,'submissions':0,
       'provider':error['error']['metadata']['provider_name'],'http_status':429,
       'provider_prompt_tokens':None,'provider_completion_tokens':None,'provider_cost_usd':None,
       'episode_wall_seconds':result['elapsed_seconds'],'failed_request_latency_seconds':None,
       'latency_limitation':'Failure attempt ledger lacks per-request timestamps. 4.128 seconds is entire episode wall, not a measured request latency.',
       'scope':'Proves bounded abort and artifact preservation, not successful integration. No usage receipt: prompt/completion/cost unknown, not zero. Original AUDIT.json contains vacuous zero sums and is not an authoritative spend report.'}
(HERE/'FAILURE-AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n'); print(json.dumps(audit)); raise SystemExit(0 if audit['passed'] else 1)
