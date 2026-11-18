#!/usr/bin/env python3
"""Read-only audit of the v2 public run against its card. Asserts what it prints."""
import json, os, re
from pathlib import Path
HERE=Path(__file__).resolve().parent; WS=HERE/'raw/qualification/solver-workspaces'
V8={'ls20':(17,63,162.5,625),'s5i5':(18,45,125.1,607),'cd82':(12,54,113.6,616)}  # actions, actor calls, critic s, wall
def rows(p): return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
out={}
for ws in sorted(WS.iterdir()):
    if not ws.is_dir(): continue
    g=ws.name.split('-')[0]; ipc=ws/'ipc'
    calls=rows(ipc/'model_calls.jsonl'); cr=[r for r in calls if r['request_purpose']=='critic']; ac=[r for r in calls if r['request_purpose']=='actor']
    lc=rows(ipc/'request_lifecycle.jsonl'); adm=[r for r in lc if r['event']=='admitted']
    prog=[p for p in rows(HERE/'raw/qualification/game-progress.jsonl') if p['game_id'].startswith(g)][0]
    wall=prog['game_elapsed_seconds']; cw=sum(r['wall_seconds'] for r in cr)
    reviews=rows(ipc/'critic_reviews.jsonl'); obj=rows(ipc/'objections.jsonl'); acts=rows(ipc/'actions.jsonl')
    executed=prog['actions']
    ratios=[]
    for i,r in enumerate(calls):
        if r['request_purpose']=='critic':
            nxt=next((x for x in calls[i+1:] if x['request_purpose']=='actor'),None)
            if nxt: ratios.append(r['prompt_tokens']/nxt['prompt_tokens'])
    pending_calls=0  # critic requests admitted while an objection pending: reconstruct by walking events
    # Walk: objection opens at a completed review with a question; closes at objections.jsonl closed/expired
    opened=[r['review_id'] for r in reviews if r['status']=='completed' and (r.get('question') or {}).get('question')]
    closed=[o['review_id'] for o in obj if o['event'] in ('closed','expired')]
    out[g]=dict(stop=prog['stop_reason'],levels=prog['levels_completed'],executed=executed,submitted=sum(len(a['actions']) for a in acts),
        actor_calls=len(ac),critic_calls=len(cr),calls_per_action=round(len(ac)/executed,1) if executed else None,
        critic_wall=round(cw,1),critic_share=round(100*cw/wall,1),critic_mean=round(cw/len(cr),1) if cr else None,
        critic_ptok_max=max((r['prompt_tokens'] for r in cr),default=0),critic_ratio_max=round(max(ratios),3) if ratios else None,
        reviews={s:sum(r['status']==s for r in reviews) for s in ('completed','invalid_output','restated')},
        empty_questions=sum(1 for r in reviews if r['status']=='completed' and not (r.get('question') or {}).get('question')),
        objections_opened=len(opened),closed=len([o for o in obj if o['event']=='closed']),expired=len([o for o in obj if o['event']=='expired']),
        dispositions=[o['disposition'] for o in obj if o['event']=='closed'],
        reminders=[r.get('reminders') for r in reviews if r['status']=='restated'],
        recurrence_rejected=len([o for o in obj if o['event']=='recurrence_rejected']),probe_exhausted=len([o for o in obj if o['event']=='probe_exhausted']),
        pending_at_end=len(opened)-len(closed),lifecycle=(len(adm),sum(r['event']=='completed' for r in lc),sum(r['event']=='failed' for r in lc)),
        done_refused=(ipc/'done_refused.jsonl').exists(),
        questions=[(r['review_id'],(r.get('question') or {}).get('question','')) for r in reviews if r['status']=='completed'],
        closes=[(o['review_id'],o['disposition'],o.get('evidence','')[:160]) for o in obj if o['event']=='closed'])
hdr=f"{'game':5}{'stop':24}{'lvl':>4}{'acts':>5}{'v8':>4}{'calls':>6}{'c/act':>6}{'v8':>5}{'crit s':>7}{'share':>7}{'v8%':>6}{'mean':>6}{'ptok':>6}{'ratio':>7}{'reviews c/i/r':>15}{'obj':>4}{'closed':>7}{'exp':>4}{'pend':>5}"
print(hdr)
for g,o in out.items():
    v=V8[g]
    print(f"{g:5}{o['stop']:24}{o['levels']:>4}{o['executed']:>5}{v[0]:>4}{o['actor_calls']:>6}{o['calls_per_action']:>6}{round(v[1]/v[0],1):>5}{o['critic_wall']:>7}{o['critic_share']:>6}%{round(100*v[2]/v[3],1):>5}%{o['critic_mean']:>6}{o['critic_ptok_max']:>6}{o['critic_ratio_max']:>7}{str(list(o['reviews'].values())):>15}{o['objections_opened']:>4}{o['closed']:>7}{o['expired']:>4}{o['pending_at_end']:>5}")
print()
for g,o in out.items():
    print(f"[{g}] dispositions={o['dispositions']} restated_reminders={o['reminders']} recurrence_rejected={o['recurrence_rejected']} probe_exhausted={o['probe_exhausted']} empty_questions={o['empty_questions']} lifecycle(adm,done,fail)={o['lifecycle']} done_refused={o['done_refused']}")
print("\n=== every question ===")
count_re=re.compile(r'\b(only|just)\s+(one|two|three|four|five|\d+)\s+(wall-clock\s+)?actions?\s+(remain|left)',re.I)
flagged=[]
for g,o in out.items():
    for rid,q in o['questions']:
        print(f"  {g} {rid}: {q or '(empty)'}")
        if count_re.search(q): flagged.append((g,rid,q))
print("\n=== every closes ===")
for g,o in out.items():
    for rid,d,e in o['closes']: print(f"  {g} {rid} {d}: {e}")
print("\n=== card checks ===")
def check(name,ok): print(('PASS ' if ok else 'FAIL ')+name); return ok
allok=True
allok&=check('all games end by budget/win, no exception',all(o['stop'] in ('model_admission_budget','wall_budget','step_budget','win') for o in out.values()) and len(out)==3)
allok&=check('no done_refused',not any(o['done_refused'] for o in out.values()))
allok&=check('critic wall share <8% every game',all(o['critic_share']<8 for o in out.values()))
allok&=check('critic mean call <8s',all(o['critic_mean']<8 for o in out.values()))
allok&=check('critic prompt <8k and <15% of actor prompt',all(o['critic_ptok_max']<8000 and o['critic_ratio_max']<0.15 for o in out.values()))
allok&=check('every objection closed/expired (none pending at end)',all(o['pending_at_end']<=0 for o in out.values()))
allok&=check('at most one expiry per game',all(o['expired']<=1 for o in out.values()))
allok&=check('no critic question states a remaining-action count as fact',not flagged)
allok&=check('all requests completed, none failed',all(o['lifecycle'][2]==0 and o['lifecycle'][0]==o['lifecycle'][1] for o in out.values()))
print('flagged questions:',flagged)
(HERE/'AUDIT.json').write_text(json.dumps(out,indent=1))
print('\nALL CARD PREDICTIONS HELD' if allok else '\nSOME PREDICTIONS FAILED')
