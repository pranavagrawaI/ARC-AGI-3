"""Descriptive token/interaction diagnostics; full phases include shutdown drain."""
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('v1_harness_audit',HERE.parent/'harness_audit.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def measure(out):
    result=audit.evictions(out);games={};errors=[];tool_results={}
    for path in sorted(out.glob('*_p0_requests.jsonl')):
        first_request=None;responses=[]
        for line in path.open():
            row=json.loads(line)
            if row.get('event')=='request' and first_request is None:
                first_request=hashlib.sha256(json.dumps(row['messages'],sort_keys=True,ensure_ascii=False).encode()).hexdigest()
            if row.get('event')=='response':
                responses.append({'tokens':(row.get('usage') or {}).get('completion_tokens',0) or 0,
                                  'reasoning_tokens':(row.get('usage') or {}).get('reasoning_tokens',0) or 0,
                                  'finish_reason':row.get('finish_reason'),'analysis_step':row.get('analysis_step')})
            for msg in row.get('messages',[]):
                if msg.get('role')!='tool' or not isinstance(msg.get('content'),str):continue
                try:tool=json.loads(msg['content'])
                except (ValueError,TypeError):continue
                if not isinstance(tool,dict) or 'returncode' not in tool:continue
                key=(path.name,msg.get('tool_call_id'),hashlib.sha256(msg['content'].encode()).hexdigest())
                if key in tool_results:continue
                tool_results[key]=tool.get('returncode')
                if tool.get('returncode'):
                    errors.append({'file':path.name,'analysis_step':row.get('analysis_step'),
                        'returncode':tool['returncode'],'stderr_excerpt':str(tool.get('stderr',''))[-450:]})
        games[path.name]={'first_request_sha256':first_request,'responses':len(responses),
                         'completion_tokens':sum(r['tokens'] for r in responses),
                         'length_cutoffs':sum(r['finish_reason']=='length' for r in responses),
                         'first_responses':responses[:3]}
    result.update(mean_tokens_per_response=result['completed_response_tokens']/result['responses'] if result['responses'] else None,
                  unique_logged_tool_result_signatures=len(tool_results),
                  nonzero_logged_tool_result_signatures=sum(bool(v) for v in tool_results.values()),
                  logged_tool_error_examples=errors[:20],games=games)
    return result


def main():
    arms={a:measure(HERE/'output/reference-mass'/a) for a in ['control','candidate']}
    matched=[]
    for name,c in arms['control']['games'].items():
        d=arms['candidate']['games'].get(name)
        if d and c['first_request_sha256']==d['first_request_sha256'] and c['first_responses'] and d['first_responses']:
            matched.append({'file':name,'control':c['first_responses'][0],'candidate':d['first_responses'][0]})
    result={'scope':'Full phase receipts including shutdown drain. Tool result signatures are deduplicated observations, not complete tool-call coverage.',
            'arms':arms,'identical_first_request_pairs':matched,
            'interpretation_limit':'Stochastic responses and later diverging trajectories; descriptive diagnostics do not identify expert capacity versus missing mass as separate causes.'}
    (HERE/'evidence/conversion-diagnostics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({a:{k:v for k,v in r.items() if k not in ['games','changed_prior_message_events','logged_tool_error_examples']} for a,r in arms.items()},indent=2))


if __name__=='__main__':main()
