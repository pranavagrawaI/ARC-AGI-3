"""Read-only extraction and accounting for real critic endurance evidence."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def rows(p):return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def audit(out):
    requests=rows(out/'requests.jsonl');responses=rows(out/'responses.jsonl')
    reviews=rows(out/'reviews.jsonl');actions=rows(out/'actions.jsonl');contexts=rows(out/'review-contexts.jsonl')
    bycall={r['call']:r for r in requests};cells=[]
    for r in responses:
        if r['purpose']!='actor':continue
        msg=r['response']['choices'][0]['message']
        code=[]
        for call in msg.get('tool_calls') or []:
            try:code.append(json.loads(call['function']['arguments']).get('code',''))
            except ValueError:code.append('[malformed tool arguments]')
        cells.append({k:r[k] for k in ('call','arm','actions_before')}|{'content':msg.get('content'),'emitted_code':code})
    timeline=[]
    for review in reviews:
        cid=review['call'];context=next(c for c in contexts if c['review_id']==review['review_id'])
        body=bycall[cid]['request']
        quoted=json.loads(body['messages'][1]['content'])['actor_request_as_evidence']
        following=next((r for r in responses if r['call']>cid and r['arm']==review['arm'] and r['purpose']=='actor'),None)
        delivered=False;response=None
        if following:
            req=bycall[following['call']]['request']
            notice=str(req['messages'][-1].get('content',''))
            delivered=any(prefix+review['review_id'] in notice for prefix in
                          ('HOST REVIEW ', 'HOST TEMPO REVIEW '))
            response=next(c for c in cells if c['call']==following['call'])
        timeline.append({'review':review,'context_equal':quoted==context['actor_request'],
            'delivered_on_next_call':delivered,'next_actor':response,
            'subsequent_host_decisions':sum(a['arm']==review['arm'] and a['step']>review['actions_before'] for a in actions)})
    metrics=[]
    for arm in ('critic','baseline'):
        rr=[r for r in responses if r['arm']==arm];aa=[a for a in actions if a['arm']==arm]
        metrics.append({'arm':arm,'calls':len(rr),'actor_calls':sum(r['purpose']=='actor' for r in rr),
            'critic_calls':sum(r['purpose']=='critic' for r in rr),
            'cost':sum(r['response'].get('usage',{}).get('cost',0) for r in rr),
            'model_seconds':sum(r['elapsed_seconds'] for r in rr),
            'critic_seconds':sum(r['elapsed_seconds'] for r in rr if r['purpose']=='critic'),
            'actions':len(aa),'action_counts':dict(Counter(a['action'] for a in aa)),
            'levels_completed':max((a['levels_completed'] for a in aa),default=0),
            'first_action_seconds':aa[0]['elapsed_seconds'] if aa else None})
    frozen=json.loads((out/'STARTED.json').read_text())
    result={'metrics':metrics,'requests':len(requests),'responses':len(responses),
        'review_statuses':dict(Counter(r['status'] for r in reviews)),
        'review_assessments':dict(Counter(r['critique']['assessment'] for r in reviews if r['status']=='completed')),
        'all_review_contexts_equal':all(t['context_equal'] for t in timeline),
        'all_valid_reviews_delivered':all(t['delivered_on_next_call'] for t in timeline if t['review']['status']=='completed'),
        'no_invalid_review_delivered':all(not t['delivered_on_next_call'] for t in timeline if t['review']['status']!='completed'),
        'frozen_files_unchanged':all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in frozen['files'].items()),
        'result_rows':json.loads((out/'RESULTS.json').read_text()) if (out/'RESULTS.json').exists() else [],
        'scope':'mechanical extraction; emitted cells are not automatically successful execution or warranted reasoning'}
    for name,value in [('AUDIT.json',result),('REVIEW-TIMELINE.json',timeline),('ACTOR-CELLS.json',cells)]:
        (out/name).write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path)
    audit(p.parse_args().output)
