"""Account for the frozen reasoning-off discovery factorial."""
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
OUT=HERE.parents[2]/'analysis/deployment-probes/20260908-discovery-broader-v2/raw/openrouter'

def rows(p):
    return [json.loads(s) for s in p.read_text().splitlines() if s.strip()] if p.exists() else []

def summarize():
    req,res=rows(OUT/'requests.jsonl'),rows(OUT/'responses.jsonl')
    errors,delivery=rows(OUT/'errors.jsonl'),rows(OUT/'delivery.jsonl')
    results=json.loads((OUT/'RESULTS.json').read_text())
    summaries=[]
    for arm in ('a','b','c','d'):
        arm_res=[r for r in res if r['arm']==arm]
        calls=[r for p in (OUT/arm).rglob('model_calls.jsonl') for r in rows(p)]
        transitions=[r for p in (OUT/arm).rglob('transitions.jsonl') for r in rows(p)]
        reasoning=[r for r in rows(OUT/'reasoning.jsonl') if r['arm']==arm]
        reasoning_off=all(not r.get('reasoning_tokens',0) and not r.get('reasoning_present') for r in reasoning)
        valid_tools=any(c.get('tool_calls') for c in calls)
        result=next((r for r in results if r['arm']==arm),{})
        ok=(reasoning_off and valid_tools and len(transitions)>=2 and len(arm_res)==len(calls)
            and result.get('error') is None and not list((OUT/arm).rglob('ABORT')))
        summaries.append(dict(arm=arm,status='PASS' if ok else 'INCOMPLETE',
            reasoning_off_verified=reasoning_off,usable_tools_observed=valid_tools,
            admitted=sum(r['arm']==arm for r in req),responses=len(arm_res),recorded_model_calls=len(calls),
            transport_errors=sum(r['arm']==arm for r in errors),
            completed_not_delivered=sum(r['arm']==arm for r in delivery),
            unresolved_attempts=[r['call'] for r in req if r['arm']==arm and
                not any(t['arm']==arm and t['call']==r['call'] for t in res+errors)],
            reasoning_tokens=sum(r.get('reasoning_tokens',0) for r in reasoning),
            completion_tokens=sum(r['response'].get('usage',{}).get('completion_tokens',0) for r in arm_res),
            actions=len(transitions),levels=max((t['after'].get('levels_completed',0) for t in transitions),default=0),
            error=result.get('error'),wall_seconds=result.get('wall_seconds')))
    return dict(status='PASS' if all(a['status']=='PASS' for a in summaries) else 'INCOMPLETE',
        arms=summaries,reported_cost=sum(r['response'].get('usage',{}).get('cost',0) or 0 for r in res),
        max_reserved=max((r.get('reserved_total',0) for r in results),default=0),
        scope='Reasoning-off two-transition screen only; no Kaggle GPU or generalized gameplay qualification.')

if __name__=='__main__':
    result=summarize();text=json.dumps(result,indent=2)+'\n'
    (OUT/'QUALIFICATION.json').write_text(text);print(text)
