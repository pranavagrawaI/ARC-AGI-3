"""Score human evidence annotations, never self-reported critique uptake labels."""
import argparse
import json
from pathlib import Path

def score_arm(events,stop):
    """Events are source-bound graded observations, sorted by elapsed wall time.

    correction/retained/false_pivot/regression refer to the predeclared commitment.
    decision is an actual host action. No critic-origin requirement exists.
    """
    times=[e['seconds'] for e in events]
    if times!=sorted(times) or any(t<0 or t>stop['seconds'] for t in times):
        raise ValueError('events must be ordered within arm window')
    corrections=[e for e in events if e['kind']=='correction' and e['warranted']]
    first=corrections[0] if corrections else None
    retained=None
    for c in corrections:
        later=[e for e in events if e['seconds']>c['seconds']]
        decisions=[e for e in later if e['kind']=='decision'][:3]
        if len(decisions)<3:continue
        until=decisions[-1]['seconds']
        window=[e for e in later if e['seconds']<=until]
        confirmations={e['decision_id'] for e in window if e['kind']=='retained' and e['warranted']}
        if all(e['decision_id'] in confirmations for e in decisions) and not any(e['kind']=='regression' and e['warranted'] for e in window):
            retained={'correction_seconds':c['seconds'],'confirmed_seconds':until,
                      'actions_to_correction':sum(e['kind']=='decision' and e['seconds']<=c['seconds'] for e in events)}
            break
    return {'first_correction_seconds':first['seconds'] if first else None,
      'actions_to_first_correction':sum(e['kind']=='decision' and first is not None and e['seconds']<=first['seconds'] for e in events) if first else None,
      'retained_correction':retained,
      'false_pivots':sum(e['kind']=='false_pivot' and e['warranted'] for e in events),
      'regressions':sum(e['kind']=='regression' and e['warranted'] for e in events),
      'followup_status':'confirmed' if retained else 'unresolved',
      'observed_seconds':stop['seconds'],'operationally_valid':stop['operationally_valid'],
      'censor_reason':None if retained else stop['reason']}

def text_fields(message):
    values=[message.get('content') or '']
    for t in message.get('tool_calls') or []:
        try:values.append(json.loads(t['function']['arguments']).get('code',''))
        except (ValueError,KeyError):pass
    return values

def load_arm(run,arm,annotations):
    responses=[json.loads(l) for l in (run/'responses.jsonl').read_text().splitlines()]
    actions=[json.loads(l) for l in (run/'actions.jsonl').read_text().splitlines()] if (run/'actions.jsonl').exists() else []
    calls={r['call']:r for r in responses if r['arm']==arm and r['purpose']=='actor'}
    aa={a['step']:a for a in actions if a['arm']==arm}
    events=[{'kind':'decision','decision_id':a['step'],'seconds':a['elapsed_seconds'],'warranted':True} for a in aa.values()]
    for a in annotations:
        if a['kind'] not in ('correction','retained','false_pivot','regression'):raise ValueError('unknown grade kind')
        if not isinstance(a['warranted'],bool) or not a.get('rationale') or not a.get('evidence_quote'):raise ValueError('grade needs judgment and evidence')
        row=calls[a['actor_call']]
        if not any(a['evidence_quote'] in s for s in text_fields(row['response']['choices'][0]['message'])):raise ValueError('unbound Actor evidence')
        event={**a,'seconds':row['arm_elapsed_seconds']}
        if a['kind']=='retained':
            action=aa[a['decision_id']]
            if event['seconds']>action['elapsed_seconds']:raise ValueError('retention evidence must precede its executed decision')
            event['seconds']=action['elapsed_seconds']
        events.append(event)
    events.sort(key=lambda e:(e['seconds'],e['kind']=='decision'))
    result=next(r for r in json.loads((run/'RESULTS.json').read_text()) if r['arm']==arm)
    scored=score_arm(events,{'seconds':result['seconds'],'operationally_valid':not result['fatal'] and result['unsettled_reservation']==0,
        'reason':result['fatal'] or result['cap'] or 'run ended'})
    rr=[r for r in responses if r['arm']==arm]
    scored.update(critic_seconds=sum(r.get('elapsed_seconds',0) for r in rr if r['purpose']=='critic'),
      critic_calls=sum(r['purpose']=='critic' for r in rr),actions=len(aa),
      confirmed_cost=sum(r['response'].get('usage',{}).get('cost',0) for r in rr),
      unresolved_cost_reservation=result['unsettled_reservation'])
    return scored

def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('grades',type=Path);args=p.parse_args()
    grades=json.loads(args.grades.read_text())
    if not grades.get('review_complete'):raise ValueError('complete the human evidence review first')
    finished={r['arm'] for r in json.loads((args.run/'RESULTS.json').read_text())}
    result={arm:load_arm(args.run,arm,grades['arms'][arm]) if arm in finished else
       {'operationally_valid':False,'retained_correction':None,'status':'missing arm; comparison unresolved'}
       for arm in ['baseline','critic']}
    # Never turn a censored/missing correction into a numeric speedup.
    b,c=result['baseline'],result['critic']
    result['seconds_saved']=b['retained_correction']['confirmed_seconds']-c['retained_correction']['confirmed_seconds'] if all(r['operationally_valid'] and r['retained_correction'] for r in (b,c)) else None
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
