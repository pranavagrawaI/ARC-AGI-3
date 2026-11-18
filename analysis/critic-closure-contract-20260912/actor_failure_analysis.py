#!/usr/bin/env python3
"""Where the Actor spends its calls and its seconds, across the three public runs."""
import json, re, sys
from collections import Counter, defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RUNS={'v8':ROOT/'analysis/deployment-probes/20260912-epistemic-questioner-public-v1',
      'v9':ROOT/'analysis/deployment-probes/20260912-objection-closure-public-v1',
      'v10':ROOT/'analysis/deployment-probes/20260912-objection-closure-public-v2'}
def rows(p): return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
def code_of(r):
    tc=r.get('tool_calls') or []
    if not tc: return ''
    try: return json.loads(tc[0]['arguments']).get('code','')
    except Exception: return tc[0]['arguments']
def classify(code):
    """Coarse, rule-based; the label is the cell's *dominant* purpose."""
    c=code
    if 'submit_actions(' in c: return 'submit'
    if re.search(r'write_helper\(|self\.h\.|load_helpers\(',c): return 'helper'
    if re.search(r'append_notes\(|write_notes\(',c): return 'notes'
    if re.search(r'plan_actions\(|set_experiment\(|preflight\(',c): return 'plan'
    if re.search(r'last_transition\(|changed_region\(|transitions?\b|trajectory\(',c): return 'inspect_transition'
    if re.search(r'grid_array\(|model_grid\(|latest_state\(|grid_rows|observations\(|decision_packet\(',c): return 'inspect_grid'
    if re.search(r'print\(',c): return 'inspect_other'
    return 'other'
def render(code,n=110):
    ls=[l for l in code.splitlines() if l.strip()]
    return (ls[0] if ls else '')[:n]
summary=[]
per_game_seq={}
for run,base in RUNS.items():
    ws=base/'raw/qualification/solver-workspaces'
    for g in sorted(p for p in ws.iterdir() if p.is_dir()):
        game=g.name.split('-')[0]; ipc=g/'ipc'
        calls=rows(ipc/'model_calls.jsonl'); ac=[r for r in calls if r['request_purpose']=='actor']
        acts=rows(ipc/'actions.jsonl'); trans=rows(ipc/'transitions.jsonl')
        labels=[classify(code_of(r)) for r in ac]
        secs=defaultdict(float); n=Counter()
        for r,l in zip(ac,labels): secs[l]+=r['wall_seconds']; n[l]+=1
        # rejections: a submit cell followed by another call in the same turn means it was REJECTED (or a non-yield)
        rejected=0
        for i,(r,l) in enumerate(zip(ac,labels)):
            if l=='submit' and i+1<len(ac) and ac[i+1]['turn']==r['turn']: rejected+=1
        executed=len(trans)
        submitted=sum(len(a['actions']) for a in acts)
        batches=[len(a['actions']) for a in acts]
        # repeated identical actions
        flat=[x for a in acts for x in a['actions']]
        rep=Counter(flat).most_common(1)[0] if flat else ('',0)
        # calls per turn distribution
        cpt=Counter(r['turn'] for r in ac)
        total=sum(r['wall_seconds'] for r in ac)
        summary.append(dict(run=run,game=game,actor_calls=len(ac),executed=executed,submitted=submitted,
            calls_per_action=round(len(ac)/executed,1) if executed else None,
            mean_call_s=round(total/len(ac),1),
            max_calls_in_one_turn=max(cpt.values()) if cpt else 0,
            submits=n['submit'],rejected_submits=rejected,
            batches_gt1=sum(b>1 for b in batches),
            most_repeated=f"{rep[0]}x{rep[1]}",
            share_calls={k:round(100*v/len(ac)) for k,v in n.most_common()},
            share_secs={k:round(100*v/total) for k,v in sorted(secs.items(),key=lambda kv:-kv[1])},
            mean_ptok=round(sum(r['prompt_tokens'] for r in ac)/len(ac)),
            mean_ctok=round(sum(r['completion_tokens'] or 0 for r in ac)/len(ac))))
        per_game_seq[(run,game)]=[(r['turn'],l,round(r['wall_seconds']),render(code_of(r))) for r,l in zip(ac,labels)]
print(f"{'run':4}{'game':5}{'calls':>6}{'acts':>5}{'c/act':>6}{'s/call':>7}{'maxc/turn':>10}{'subm':>5}{'rej':>4}{'batch>1':>8}{'ptok':>7}{'ctok':>6}  most_repeated  calls% / secs%")
for s in summary:
    print(f"{s['run']:4}{s['game']:5}{s['actor_calls']:>6}{s['executed']:>5}{s['calls_per_action']:>6}{s['mean_call_s']:>7}{s['max_calls_in_one_turn']:>10}{s['submits']:>5}{s['rejected_submits']:>4}{s['batches_gt1']:>8}{s['mean_ptok']:>7}{s['mean_ctok']:>6}  {s['most_repeated']:14} {s['share_calls']}")
    print(f"{'':46}secs: {s['share_secs']}")
if '--seq' in sys.argv:
    key=tuple(sys.argv[sys.argv.index('--seq')+1].split(':'))
    for t,l,w,c in per_game_seq[key]: print(f"  t{t:<3}{l:19}{w:>4}s  {c}")
json.dump(summary,open(Path(__file__).with_name('actor_failure_summary.json'),'w'),indent=1)
