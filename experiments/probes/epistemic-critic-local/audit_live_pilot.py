"""Recompute transport/context/uptake evidence; no semantic grades or inference."""
import argparse
from collections import Counter
import json
from pathlib import Path


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def audit(out):
    requests=rows(out/'requests.jsonl');responses=rows(out/'responses.jsonl')
    contexts=rows(out/'review_contexts.jsonl');actions=rows(out/'actions.jsonl')
    metrics=[]
    for row in responses:
        usage=row['response'].get('usage',{})
        metrics.append({**{k:row[k] for k in ('call','arm','purpose','elapsed_seconds')},
            **{k:usage.get(k) for k in ('prompt_tokens','completion_tokens','cost')},
            'cached_tokens':usage.get('prompt_tokens_details',{}).get('cached_tokens'),
            'reasoning_tokens':usage.get('completion_tokens_details',{}).get('reasoning_tokens')})
    critic=[r for r in requests if r['purpose']=='critic']
    prefix=[]
    for r in critic:
        context=next(c['request'] for c in contexts if c['arm']==r['arm'])
        quoted=None
        try:
            quoted=json.loads(r['request']['messages'][1]['content']).get('actor_request_as_evidence')
        except (ValueError,AttributeError): pass
        prefix.append({'call':r['call'],
            'messages_equal':r['request']['messages'][:-1]==context['messages'],
            'tools_equal':r['request'].get('tools')==context.get('tools'),
            'tool_choice_none':r['request'].get('tool_choice')=='none',
            'quoted_context_equal':quoted==context,
            'no_declared_tools':not r['request'].get('tools')})
    codes=[]
    for r in responses:
        if r['purpose']!='actor':continue
        message=r['response']['choices'][0]['message']
        for call in message.get('tool_calls') or []:
            fn=call['function']
            codes.append({'call':r['call'],'arm':r['arm'],'tool':fn['name'],
                          'arguments':json.loads(fn['arguments'])})
    result={'requests':len(requests),'responses':len(responses),
        'actual_cost':sum(m['cost'] or 0 for m in metrics),
        'reservation':sum(r['reservation'] for r in requests),
        'requests_by_arm':dict(Counter(r['arm'] for r in requests)),
        'actions_by_arm':dict(Counter(r['arm'] for r in actions)),
        'critic_prefix_checks':prefix,'request_metrics':metrics,
        'critiques':rows(out/'critiques.jsonl'),
        'stop_records':rows(out/'errors.jsonl'),
        'observed_actions':[{'arm':a['arm'],'action':a['action'],
            'elapsed_seconds':a['elapsed_seconds'],'levels_completed':a['levels_completed'],
            'changed_cells':sum(x!=y for r,s in zip(a['before'],a['after']) for x,y in zip(r,s))}
            for a in actions],
        'first_two_actions_same_transition':(
            actions[0]['before']==actions[1]['before'] and actions[0]['after']==actions[1]['after']
            if len(actions)>=2 else None),
        'results':json.loads((out/'RESULTS.json').read_text()) if (out/'RESULTS.json').exists() else None,
        'scope':'mechanical audit only; explicit Actor outputs require human epistemic grading'}
    (out/'AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'ACTOR-CELLS.json').write_text(json.dumps(codes,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path)
    result=audit(parser.parse_args().output)
    print(json.dumps({k:v for k,v in result.items() if k not in ('request_metrics','critiques')},indent=2))
