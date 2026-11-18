"""Mechanical audit of the reasoning screen; manual semantic judgments are separate."""
import argparse
import hashlib
import json
from pathlib import Path

from claim_screen import decode_json
from ledger_screen import BASELINE


def audit(out):
    cases=json.loads((out/'CASES.json').read_text())
    frozen=json.loads((out/'STARTED.json').read_text())
    prepared=json.loads((out/'PREPARED.json').read_text())
    assert hashlib.sha256((out/'EXPERIMENT-CARD.md').read_bytes()).hexdigest()==frozen['card_sha256']
    # Verify factual premises against already source-verified saved prefix records.
    baseline=json.loads((BASELINE/'cases.json').read_text())
    assert len(baseline[0]['evidence'])==3 and len(baseline[4]['evidence'])==1
    assert len({t['action'] for t in baseline[0]['evidence']})==3
    for idx in (0,4):
        for t in baseline[idx]['evidence']:
            assert {c[0] for c in t['changes_y_x_before_after']}=={63}
            assert t['levels_before']==t['levels_after']==0
    assert baseline[4]['evidence'][0]['action']=='UP'
    assert len(baseline[4]['evidence'][0]['changes_y_x_before_after'])==1
    result={'calls':0,'label_matches':0,'cost_usd':0,'seconds':0,'prompt_tokens':0,
            'completion_tokens':0,'format_findings':[],'cases':[]}
    for i,c in enumerate(cases):
        file=out/f'{i:02d}.request.json'
        assert hashlib.sha256(file.read_bytes()).hexdigest()==frozen['request_sha256'][file.name]
        body=json.loads(file.read_text())
        assert hashlib.sha256(body['messages'][0]['content'].encode()).hexdigest()==prepared['system_sha256']
        packet=json.loads(body['messages'][1]['content'])
        assert packet=={'stipulated_observations':c['observations'],'actor_reasoning':c['actor_reasoning']}
        response=json.loads((out/f'{i:02d}.response.json').read_text())
        usage=response['usage'];choice=response['choices'][0]
        assert response['provider']=='CoreWeave' and choice['finish_reason']=='stop'
        assert usage['completion_tokens_details']['reasoning_tokens']==0
        content=decode_json(choice['message']['content'])
        assert set(content)=={'assessment','actor_quote','inference_gap','decision_consequence','minimal_revision'}
        assert content['assessment'] in ('concern','no_concern','unclear')
        assert content['actor_quote'] in c['actor_reasoning']
        count=sum(len(content[k].split()) for k in ('inference_gap','decision_consequence','minimal_revision'))
        if len(content['actor_quote'].split())>25 or count>90:
            result['format_findings'].append({'case':c['id'],'issue':'word limit'})
        result['calls']+=1;result['label_matches']+=content['assessment']==c['expected']
        result['cost_usd']+=usage['cost'];result['prompt_tokens']+=usage['prompt_tokens']
        result['completion_tokens']+=usage['completion_tokens']
        result['seconds']+=json.loads((out/f'{i:02d}.timing.json').read_text())['seconds']
        result['cases'].append({'id':c['id'],'expected':c['expected'],'observed':content['assessment']})
    for a,b in zip(cases[::2],cases[1::2]):
        assert a['observations']==b['observations'] and a['pair']==b['pair']
    execution=json.loads((out/'EXECUTION.json').read_text())
    assert execution['status']=='completed' and execution['attempted']==8
    assert abs(execution['actual_cost_usd']-result['cost_usd'])<1e-10
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=audit(args.output)
    (args.output/'AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
