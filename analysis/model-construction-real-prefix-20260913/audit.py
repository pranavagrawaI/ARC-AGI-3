#!/usr/bin/env python3
"""Read-only episode audit; emits AUDIT.json without rerunning model/evaluation."""
import hashlib
import json
from decimal import Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent

def read(name):
    return json.loads((HERE / name).read_text())

def lines(name):
    return [json.loads(line) for line in (HERE / name).read_text().splitlines()]

result = read('RESULT.json')
started = read('STARTED.json')
pre = read('PRELAUNCH.json')
requests = [json.loads(p.read_text()) for p in sorted((HERE/'raw/requests').glob('*.json'))]
responses = [json.loads(p.read_text()) for p in sorted((HERE/'raw/responses').glob('*.json'))]
events = lines('raw/tool_events.jsonl')
attempts = lines('raw/request_attempts.jsonl')
calls = lines('raw/model_calls.jsonl')
transcript = read('raw/transcript.json')
checks = {
    'prelaunch_passed': pre['passed'] is True,
    'source_and_card_unchanged_since_start': all(hashlib.sha256((HERE/name).read_bytes()).hexdigest() == digest for name,digest in started['artifact_sha256'].items()),
    'request_response_counts_reconcile': len(requests) == len(responses) == len(calls) == result['model_requests'] == 4,
    'one_attempt_per_request_no_failed_or_retried_request': [x['index'] for x in attempts if x['status']=='started'] == list(range(4)) and [x['index'] for x in attempts if x['status']=='completed'] == list(range(4)) and len(attempts)==8,
    'model_unchanged': all(x['model']=='qwen/qwen3.8-27b' for x in requests+responses),
    'reasoning_disabled_and_zero_reported': all(x['reasoning']=={'enabled': False} for x in requests) and all(x['usage']['completion_tokens_details']['reasoning_tokens']==0 for x in responses),
    'request_token_cap_reconciles': sum(x['max_tokens'] for x in requests)==result['requested_tokens']==8192,
    'prompt_usage_reconciles': sum(x['usage']['prompt_tokens'] for x in responses)==result['usage']['prompt_tokens'],
    'completion_usage_reconciles': sum(x['usage']['completion_tokens'] for x in responses)==result['usage']['completion_tokens'],
    'durable_transcript_matches_requests_and_responses': transcript==[{'request': a, 'response': b} for a,b in zip(requests,responses)],
    'responses_match_raw_bodies': all(json.loads((HERE/'raw/responses'/f'{i:02d}.body').read_bytes())==response for i,response in enumerate(responses)),
    'call_artifact_hashes_match': all(hashlib.sha256((HERE/c[k+'_path']).read_bytes()).hexdigest()==c[k+'_sha256'] for c in calls for k in ('request','response')),
    'tool_calls_exact_and_successful': len(events)==result['inspections']==4 and all(event['call']==responses[i]['choices'][0]['message']['tool_calls'][0] and json.loads(event['rendered_result_message']['content'])['status']=='completed' for i,event in enumerate(events)),
    'request_snapshots_have_no_later_messages': [len(x['messages']) for x in requests]==[2,4,6,8],
    'tool_results_exact_in_next_request': all(requests[i+1]['messages'][-1]==events[i]['rendered_result_message'] for i in range(len(requests)-1)),
    'suffix_hash_never_in_requests': all(started['sealed_suffix_sha256'] not in json.dumps(x) for x in requests),
    'no_candidate_or_suffix_evaluation': result['submissions']==0 and result['sealed_suffix_evaluation'] is None and not (HERE/'raw/FROZEN.json').exists() and not (HERE/'raw/sealed_suffix_evaluation.json').exists(),
    'wall_cap_respected': result['elapsed_seconds'] < 600,
}
audit = {
    'passed': all(checks.values()), 'checks': checks,
    'model_requests': len(requests), 'inspections': len(events),
    'submissions': result['submissions'],
    'requested_completion_tokens': sum(x['max_tokens'] for x in requests),
    'reported_prompt_tokens': sum(x['usage']['prompt_tokens'] for x in responses),
    'reported_completion_tokens': sum(x['usage']['completion_tokens'] for x in responses),
    'provider_reported_cost_usd': str(sum(Decimal(str(x['usage']['cost'])) for x in responses)),
    'scope': 'Artifact consistency and bounded episode audit, not a new experiment or competence result',
}
(HERE/'AUDIT.json').write_text(json.dumps(audit, indent=2, sort_keys=True)+'\n')
print(json.dumps(audit, sort_keys=True))
assert audit['passed']
