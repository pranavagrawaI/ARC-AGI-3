"""Summarize observed solver behavior without equating encoder agreement to truth."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def summarize(root):
    sessions=[]
    hashes={}
    for session in sorted((root/'solver-workspaces').glob('*')):
        if not session.is_dir():continue
        ipc=session/'ipc'
        transitions=rows(ipc/'transitions.jsonl')
        actions=rows(ipc/'actions.jsonl')
        calls=rows(ipc/'model_calls.jsonl')
        commits=rows(ipc/'model_predictions.jsonl')
        feedback=rows(ipc/'model_feedback.jsonl')
        latest={}
        for r in feedback:latest[(r.get('turn'),r.get('start_step'))]=r
        evaluated=sorted(latest.values(),key=lambda r:r.get('start_step',-1))
        fields=Counter(k for r in evaluated for d in r.get('differences',[]) for k in d['fields'])
        corrections=[]
        last_mismatch=None
        for r in evaluated:
            if r['status']=='mismatch':last_mismatch=r
            elif r['status']=='matched' and last_mismatch and r['model_sha256']!=last_mismatch['model_sha256']:
                corrections.append({'contradiction_step':last_mismatch['start_step'],
                                    'successful_new_version_step':r['start_step'],
                                    'old_version':last_mismatch['model_sha256'],'new_version':r['model_sha256']})
                last_mismatch=None
        errors=[]
        for c in commits:
            source=c.get('source')
            if source is not None and hashlib.sha256(source.encode()).hexdigest()!=c.get('model_sha256'):
                errors.append('frozen source hash mismatch')
        for a in actions:
            matching=[c for c in commits if c['turn']==a['turn'] and c['step']==a['step'] and c['actions']==a['actions']]
            if len(matching)!=1:errors.append(f"action turn {a['turn']} lacks unique prospective record")
        for name in ['transitions.jsonl','actions.jsonl','model_calls.jsonl','model_predictions.jsonl','model_feedback.jsonl']:
            path=ipc/name
            if path.exists():hashes[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
        sessions.append({'session':session.name,'actions':len(transitions),'actor_batches':len(actions),
                         'levels_completed':max((t['after'].get('levels_completed',0) for t in transitions),default=0),
                         'model_calls':len(calls),'summarizer_calls':sum(bool(c.get('summarizer')) for c in calls),
                         'model_wall_seconds':sum(c.get('wall_seconds',0) for c in calls),
                         'prediction_commits':len(commits),
                         'helper_versions':len({c['model_sha256'] for c in commits if c.get('model_sha256')}),
                         'latest_online_feedback_by_commit':dict(Counter(r['status'] for r in evaluated)),
                         'checked_transitions':sum(r.get('checked',0) for r in evaluated),
                         'field_mismatches':dict(fields),'revised_versions_followed_by_match':corrections,
                         'prediction_evidence_errors':errors,
                         'limits':'Source revision followed by encoder agreement is not proof of semantic correction or improved game solving.'})
    return {'schema':'q6-long-behavior-v1','sessions':sessions,'input_sha256':hashes,
            'game_stops':rows(root/'solver-workspaces/game-stops.jsonl'),
            'comparison':'none; descriptive sustained gameplay',
            'final_feedback':'Separate post-run replay files are not counted as feedback delivered to the Actor.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('raw',type=Path);p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    report=summarize(args.raw)
    (args.output_dir/'REPORT-LONG.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# q6 longer-run observations','', 'Descriptive run; no matched gameplay control.','']
    for s in report['sessions']:
        lines += [f"Session {s['session']}: {s['actions']} actions, {s['actor_batches']} Actor batches, {s['levels_completed']} levels completed.",
                  f"Model calls: {s['model_calls']}; summarizer calls: {s['summarizer_calls']}; recorded model time: {s['model_wall_seconds']:.1f}s.",
                  f"Online prediction feedback: {s['latest_online_feedback_by_commit']}; checked transitions: {s['checked_transitions']}.",
                  f"Helper versions: {s['helper_versions']}; revised versions followed by a match: {len(s['revised_versions_followed_by_match'])}.",
                  f"Field mismatches: {s['field_mismatches']}.",
                  f"Prediction evidence errors: {s['prediction_evidence_errors']}.",'']
    lines += ['Consult SCORE.json for host validity before interpreting these counts.',
              'Encoder agreement is model consistency evidence, not independent game truth. Final post-run replay was not seen by the Actor.','']
    (args.output_dir/'RESULT-LONG.md').write_text('\n'.join(lines))
    print(json.dumps(report,indent=2))
