import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
roots=[HERE,HERE.parent/'20260906-openrouter-action-probe']
summary={'runs':[],'total_reported_cost':0,'total_calls':0,'reasoning_tokens':0}
for root in roots:
    responses=[json.loads(l) for l in (root/'responses.jsonl').read_text().splitlines()]
    results=json.loads((root/'RESULTS.json').read_text())
    for result in results:
        selected=[r for r in responses if r['arm']==result['arm']]
        cost=sum(r['response']['usage']['cost'] for r in selected)
        reasoning=sum(r['response']['usage'].get('completion_tokens_details',{}).get('reasoning_tokens',0) for r in selected)
        summary['runs'].append({**result,'directory':str(root),'reported_cost':cost,'reasoning_tokens':reasoning,
            'providers':sorted({r['response'].get('provider') for r in selected})})
        summary['total_reported_cost']+=cost;summary['total_calls']+=len(selected);summary['reasoning_tokens']+=reasoning
p=roots[1]
transition=json.loads(next(p.rglob('transitions.jsonl')).read_text().splitlines()[0])
a,b=transition['before']['grid'],transition['after']['grid']
diff=[(y,x) for y in range(len(a)) for x in range(len(a[y])) if a[y][x]!=b[y][x]]
summary['action_evidence']={'action':'UP','changed_cells':len(diff),
    'actual_bbox_rows_cols':[min(y for y,x in diff),max(y for y,x in diff),min(x for y,x in diff),max(x for y,x in diff)],
    'predicted_bbox_rows_cols':[54,61,2,9],
    'changes_inside_predicted_bbox':sum(54<=y<=61 and 2<=x<=9 for y,x in diff),
    'host_retrodiction':json.loads(next(p.rglob('observations.json')).read_text())['last_retrodiction']}
(HERE/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
