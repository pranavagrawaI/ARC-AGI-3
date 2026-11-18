"""Audit paired inputs, coverage and observed labels; semantic reasons need review."""
import argparse
import hashlib
import json
from pathlib import Path

from claim_screen import decode_json


def audit(out):
    data=json.loads((out/'DATA.json').read_text())
    started=json.loads((out/'STARTED.json').read_text())
    assert hashlib.sha256((out/'EXPERIMENT-CARD.md').read_bytes()).hexdigest()==started['card_sha256']
    for name,sha in started['input_hashes'].items():
        assert hashlib.sha256((out/name).read_bytes()).hexdigest()==sha, name
    result={'calls':0,'cost_usd':0,'seconds':0,'prompt_tokens':0,'completion_tokens':0,
            'reasoning_tokens':0,'fenced_responses':0,'conditions':{},'format_findings':[]}
    replies={}
    for file in sorted(out.glob('*.response.json')):
        name=file.name.removesuffix('.response.json')
        response=json.loads(file.read_text());usage=response['usage'];choice=response['choices'][0]
        assert response['provider']=='CoreWeave' and choice['finish_reason']=='stop'
        assert usage['completion_tokens_details']['reasoning_tokens']==0
        content=choice['message']['content'];replies[name]=decode_json(content)
        timing=json.loads((out/f'{name}.timing.json').read_text())
        result['calls']+=1;result['cost_usd']+=usage['cost'];result['seconds']+=timing['seconds']
        result['prompt_tokens']+=usage['prompt_tokens'];result['completion_tokens']+=usage['completion_tokens']
        result['fenced_responses']+=content.strip().startswith('```')
    for d in data:
        prefix=d['id'];entry={'single':[],'list':[]}
        batch_input=json.loads(json.loads((out/f'{prefix}-list.request.json').read_text())['messages'][1]['content'])
        assert batch_input['claims']==d['claims'] and batch_input['facts']==d['facts']
        batch=replies[f'{prefix}-list']['judgments']
        assert [r['id'] for r in batch]==[c['id'] for c in d['claims']]
        for c in d['claims']:
            name=f'{prefix}-single-{c["id"]}'
            packet=json.loads(json.loads((out/f'{name}.request.json').read_text())['messages'][1]['content'])
            assert packet=={'facts':d['facts'],'claims':[c]}
            reply=replies[name]['judgments'];assert len(reply)==1 and reply[0]['id']==c['id']
            entry['single'].append({'claim':c['id'],'expected':d['expected'][c['id']],
                'observed':reply[0]['status'],'reason':reply[0]['reason']})
            r=next(r for r in batch if r['id']==c['id'])
            entry['list'].append({'claim':c['id'],'expected':d['expected'][c['id']],
                'observed':r['status'],'reason':r['reason']})
        extraction=replies[f'{prefix}-extract']['claims']
        assert all(c['text'] in d['narrative'] for c in extraction)
        verify_packet=json.loads(json.loads((out/f'{prefix}-narrative-verify.request.json').read_text())['messages'][1]['content'])
        assert verify_packet=={'facts':d['facts'],'claims':extraction}
        verified=replies[f'{prefix}-narrative-verify']['judgments']
        assert [r['id'] for r in verified]==[c['id'] for c in extraction]
        entry['extracted_count']=len(extraction)
        entry['extraction_targets']=[{'claim':c['id'],
            'containing_excerpts':[x['id'] for x in extraction if c['text'] in x['text']]} for c in d['claims']]
        entry['narrative_judgments']=verified
        refs={t['ref'] for t in d['facts']['transitions']}|{t['ref'] for t in d['facts']['current_color_measurements']}|{t['ref'] for t in d['facts']['current_changed_rows']}
        for r in verified:
            if set(r['evidence_refs'])-refs:
                result['format_findings'].append({'case':prefix,'claim':r['id'],
                    'unknown_ids':list(set(r['evidence_refs'])-refs)})
        result['conditions'][prefix]=entry
    assert result['calls']==12
    prior=Path(started['continued_from'])
    initial=next(prior.glob('*.response.json'))
    assert initial.read_bytes()==(out/initial.name).read_bytes()
    execution=json.loads((out/'EXECUTION.json').read_text())
    assert execution['reused_calls']==1 and execution['calls']==12
    assert abs(execution['cost_usd']-result['cost_usd'])<1e-10
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=audit(args.output)
    (args.output/'AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='conditions'},indent=2))
