"""Saved-trace measurements only. No model calls or generated-helper execution."""
from collections import Counter
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE/'raw/qualification'
def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []

def inspect():
    facts={}
    for session in sorted((ROOT/'solver-workspaces').iterdir()):
        if not session.is_dir(): continue
        ipc=session/'ipc'
        calls=rows(ipc/'model_calls.jsonl')
        transitions=rows(ipc/'transitions.jsonl')
        searches=rows(ipc/'search_results.jsonl')
        actions=rows(ipc/'actions.jsonl')
        for search in searches:
            if search['status']=='path_found':
                assert search['replay']=='matched' and search['child_reaped']
                assert hashlib.sha256(search['model_source'].encode()).hexdigest()==search['snapshot']['model_sha256']
                assert 'predicted_path' not in search
            if search['status']=='limited':
                assert not search.get('plan') and not search['complete']
        search_by_id={r['search_id']:r for r in searches if r.get('search_id')}
        for action in actions:
            if action.get('search_id'):
                r=search_by_id[action['search_id']]
                assert action['turn']==r['snapshot']['turn'] and action['step']==r['snapshot']['step']
                assert action['actions']==r['plan'][:len(action['actions'])]
        changes=[]
        for t in transitions:
            a,b=t['before']['grid'],t['after']['grid']
            same=len(a)==len(b) and all(len(x)==len(y) for x,y in zip(a,b))
            changed=[(i,j) for i,(x,y) in enumerate(zip(a,b)) for j,(u,v) in enumerate(zip(x,y)) if u!=v] if same else None
            changes.append(dict(step=t['step'],action=t['action'],
                changed_rows=sorted({i for i,j in changed}) if changed is not None else None,
                changed_cells=len(changed) if changed is not None else None))
        facts[session.name]=dict(calls=len(calls),actions=len(transitions),
            calls_per_action=len(calls)/len(transitions) if transitions else None,
            calls_by_step=dict(Counter(c['step'] for c in calls)),
            action_counts=dict(Counter(t['action'] for t in transitions)),
            search_statuses=dict(Counter(r['status'] for r in searches)),
            search_seconds=sum(r['elapsed_seconds'] for r in searches),
            linked_submissions=sum(bool(a.get('search_id')) for a in actions),
            batch_lengths=dict(Counter(len(a['actions']) for a in actions)),
            rejections=dict(Counter(r.get('reason',r.get('gate','unknown')) for r in rows(ipc/'rejections.jsonl'))),
            changes=changes)
        # Preserve Actor output without grids; use this for targeted qualitative review.
        lines=[]
        for c in calls:
            lines.append(f"\nCALL {c['request_id']} turn={c['turn']} step={c['step']} seconds={c['wall_seconds']}\n")
            if c.get('response_content'): lines.append(c['response_content'])
            for tool in c.get('tool_calls',[]):
                lines.append(tool.get('name','')+' '+str(tool.get('arguments','')))
        for a in actions:
            lines.append(f"\nSUBMITTED turn={a['turn']} step={a['step']} {a['actions']}: {a.get('rationale','')}")
        (HERE/(session.name+'-actor.txt')).write_text('\n'.join(lines))
    return dict(scope='descriptive measured behavior; semantic interpretation requires trace review',games=facts)

if __name__=='__main__':
    print(json.dumps(inspect(),indent=2))
