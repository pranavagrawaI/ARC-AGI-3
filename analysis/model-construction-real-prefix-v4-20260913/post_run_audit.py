#!/usr/bin/env python3
"""Postfreeze failure-specific audit. Read artifacts; no network or evaluator calls.

Does not modify the frozen runner/card or replace its failed original audit.
The continuation is an OFFLINE RECONSTRUCTION, not an actual wire request.
"""
import copy
import hashlib
import json
from decimal import Decimal
from pathlib import Path
from request_contract import validate_request

HERE=Path(__file__).resolve().parent
load=lambda name:json.loads((HERE/name).read_text())
digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
def lines(name):
    path=HERE/name
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def main():
    result=load('RESULT.json'); original_audit=load('AUDIT.json'); started=load('STARTED.json')
    req_paths=sorted((HERE/'raw/requests').glob('*.json'))
    res_paths=sorted((HERE/'raw/responses').glob('*.json'))
    requests=[json.loads(p.read_text()) for p in req_paths]
    responses=[json.loads(p.read_text()) for p in res_paths]
    attempts=lines('raw/request_attempts.jsonl'); events=lines('raw/tool_events.jsonl')
    provider=load('PROVIDER-SELECTION.json')['selected']['provider_name']
    prompt=sum(x['usage']['prompt_tokens'] for x in responses)
    completion=sum(x['usage']['completion_tokens'] for x in responses)
    cost=sum((Decimal(str(x['usage']['cost'])) for x in responses),Decimal(0))
    # Isolate the blocker without making another call, inspection or evaluation.
    continuation=copy.deepcopy(requests[0])
    continuation['messages']=continuation['messages'][:-1]+[copy.deepcopy(responses[0]['choices'][0]['message'])]+[e['rendered_result_message'] for e in events]+continuation['messages'][-1:]
    # Reuses the initial telemetry solely for a schema diagnostic. It is not a
    # claim about the exact unsent request or continuation budget state.
    rejected=None
    try:
        validate_request(continuation)
    except ValueError as error:
        rejected=str(error)
    normalized=copy.deepcopy(continuation)
    removed=[]
    for message in normalized['messages']:
        for call in message.get('tool_calls') or []:
            if 'index' in call:
                removed.append({'tool_call_id':call['id'],'field':'index','value':call.pop('index')})
    normalization_pass=validate_request(normalized)['passed']
    unexpected_failed={n for n,passed in original_audit['checks'].items() if not passed}
    checks={
        'frozen_files_unchanged':all(digest(HERE/n)==h for n,h in started['artifact_sha256'].items()),
        'prior_directories_unchanged':all(digest(HERE.parent/sibling/n)==h for manifest,sibling in [('BASELINE-SHA256.json','model-construction-real-prefix-20260913'),('V2-SHA256.json','model-construction-real-prefix-v2-20260913'),('V3-SHA256.json','model-construction-real-prefix-v3-20260913')] for n,h in load(manifest).items()),
        'one_actual_attempt_and_one_receipt':len(requests)==len(responses)==1 and [a['status'] for a in attempts]==['started','completed'],
        'no_network_failure_no_unknown_attempt':not [a for a in attempts if a['status']=='failed'] and not list((HERE/'raw/responses').glob('*.error.body')),
        'actual_wire_and_response_preserved':all(json.loads(p.with_suffix('.body').read_bytes())==x for p,x in zip(req_paths+res_paths,requests+responses)),
        'first_outgoing_contract_valid':validate_request(requests[0])['passed'],
        'provider_and_model_match':all(x.get('provider')==provider and x['model']=='qwen/qwen3.8-27b' for x in responses),
        'reasoning_off_in_request_and_receipt':requests[0]['reasoning']=={'enabled':False} and all(x['usage']['completion_tokens_details']['reasoning_tokens']==0 and not any(x['choices'][0]['message'].get(k) for k in ('reasoning','reasoning_content','reasoning_details')) for x in responses),
        'tokens_cost_and_wall_bounded':prompt<=300000 and completion<=16384 and cost<=Decimal('0.50') and result['elapsed_seconds']<900,
        'reservations_not_charged_as_actual':requests[0]['max_tokens']==4096 and completion==78 and original_audit['completion_tokens']==completion and original_audit['prompt_tokens']==prompt and Decimal(original_audit['provider_reported_cost_usd'])==cost,
        'one_successful_inspection_no_candidate':len(events)==1 and events[0]['call']['function']['name']=='execute_python' and json.loads(events[0]['rendered_result_message']['content'])['status']=='completed',
        'tool_call_exactly_from_response':events[0]['call']==responses[0]['choices'][0]['message']['tool_calls'][0],
        'no_suffix_or_candidate_artifacts':not any((HERE/'raw'/n).exists() for n in ('FROZEN.json','frozen_candidate.py','sealed_suffix_evaluation.json','suffix_attempts.jsonl')),
        'original_invalid_result_preserved':result['classification']=='INTEGRATION_INVALID' and result['error']=='ValueError: request contract: tool call envelope',
        'offline_diagnosis_reproduces_exact_error':rejected=='request contract: tool call envelope',
        'offline_removing_only_response_index_passes_schema':normalization_pass and len(removed)==1 and removed[0]['field']=='index',
        'original_audit_failures_explicitly_retained':original_audit['passed'] is False and unexpected_failed=={'usage_reconciles','request_allowance_reconciles','exact_transcript'},
        'no_transcript_completeness_claim':not (HERE/'raw/transcript.json').exists() and not (HERE/'raw/construction.json').exists(),
    }
    report={
        'schema':'v4-post-run-artifact-audit-v1','passed':all(checks.values()),'checks':checks,
        'classification':'INTEGRATION_INVALID_LOCAL_CONTINUATION_CONTRACT',
        'provider_parameter_compatibility':'ESTABLISHED_FOR_FIRST_ACTUAL_REQUEST_ONLY',
        'actual_provider_matches_pin':'ESTABLISHED','provider':provider,
        'actual_reasoning':'0 provider-reported reasoning tokens; null message.reasoning',
        'request_attempts':len(requests),'completed_responses':len(responses),
        'inspections':len(events),'submissions':0,'work_item':None,'candidate':None,'visible_candidate_checks':'NOT_RUN','suffix':'NOT_RUN',
        'prompt_tokens':prompt,'completion_tokens':completion,'provider_reported_cost_usd':str(cost),
        'unknown_cost_attempts':0,'receipt_status':'COMPLETE_FOR_ALL_ACTUAL_REQUESTS',
        'completion_allowance_reserved':sum(q['max_tokens'] for q in requests),
        'prompt_reservation':len(json.dumps(requests[0],ensure_ascii=False).encode())+4096,
        'request_latency_seconds':attempts[-1]['wall_seconds'],'inspection_seconds':events[0]['wall_seconds'],'episode_wall_seconds':result['elapsed_seconds'],
        'model_selected_operation':'Inspect visible case keys, initial/current grid dimensions and first visible transition; no structured work item submitted.',
        'local_failure':result['error'],'offending_response_fields':removed,
        'offline_schema_diagnostic':{'paid_calls':0,'new_inspections':0,'new_suffix_evaluations':0,'original_rejected':rejected,'index_removed_copy_passes':normalization_pass,'limitation':'Reconstructed messages using original initial telemetry; not exact unsent wire request. No repaired episode launched.'},
        'original_audit_passed':False,'original_audit_failed_checks':sorted(unexpected_failed),
        'runner_accounting_limitation':'Generic exception handler incorrectly labels local pre-send failure as unknown failed provider attempt and omits requested allowance, transcript and construction summary. All actual provider attempts have receipts; durable raw request/response/tool ledgers establish known usage above.',
        'scope':'Post-run failure-specific evidence consistency only; original integration and original audit remain failed. No candidate or capability claim.',
    }
    (HERE/'POST-RUN-AUDIT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    raise SystemExit(0 if report['passed'] else 1)

if __name__=='__main__':main()
