"""Bounded claim selection/verification experiment using two saved prefixes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

from ledger_screen import ROOT, BASELINE, read_prefix, measure

VERIFY = '''Check EVERY supplied claim separately against the supplied host facts.
Keep quantifiers and scope. Distinguish contradicted (a supplied fact conflicts),
supported_in_scope (the claim follows within its stated conditions), and unresolved
(facts neither establish nor refute it). Lack of completion need not refute an
objective hypothesis. Do not invent measurements or outcomes. Return JSON only:
{"judgments":[{"id":"supplied claim ID","status":"contradicted|supported_in_scope|unresolved",
"evidence_refs":["supplied fact ID"],"reason":"at most 25 words"}]}.
Return exactly one judgment per claim, including correct and unresolved claims.'''
EXTRACT = '''Extract independently checkable claims from this Actor narrative.
Preserve original wording and quantifiers in short exact excerpts. Include factual
counts, claimed general rules and objective hypotheses even when labelled hypotheses.
Merge repetitions but do not omit a distinct claim just because it seems minor.
Do not judge, correct or add claims. Do not include instructions to take an action.
Return JSON only: {"claims":[{"id":"c1","text":"exact contiguous excerpt"},...]}.
Use at most 12 claims. You do not have the evidence; this is extraction only.'''


def datasets():
    old = json.loads((BASELINE / 'cases.json').read_text())
    result = []
    targets = [
        (0, [('one', 'action = any CLICK x y -> exactly one bottom-row cell (next from right) turns 3->4.', 'contradicted'),
             ('four', 'Count now 4 (cols 63,62,61,60).', 'supported_in_scope'),
             ('goal', 'level completes when counter reaches threshold.', 'unresolved')]),
        (4, [('red', 'Red(2) cells: 30 cells, rows 24-32, cols 25-38', 'supported_in_scope'),
             ('white', 'White(f) region: rows 3-31, cols 3-44 (large area)', 'supported_in_scope'),
             ('indicator', '(63,63) changed from blue to gray — might be a step counter or player indicator', 'unresolved')])]
    for index, claims in targets:
        case = old[index]
        path = ROOT / case['provenance']['source']
        if path.is_dir(): path = path / 'ipc/transitions.jsonl'
        records = read_prefix(path, case['provenance']['cutoff_step'])
        measured = measure(records)
        grid = records[-1]['after']['grid']
        colors = []
        for color in sorted({c for row in grid for c in row}):
            coords = [(y,x) for y,row in enumerate(grid) for x,c in enumerate(row) if c == color]
            colors.append({'ref': f'color:{color}', 'color': color, 'cell_count': len(coords),
                           'row_bounds': [min(y for y,x in coords), max(y for y,x in coords)],
                           'column_bounds': [min(x for y,x in coords), max(x for y,x in coords)]})
        facts = {'transitions': measured['ledger'], 'current_color_measurements': colors,
                 'coverage': measured['coverage'], 'note': 'Color bounds cover all cells of that color; object identity and objectives are not measured.'}
        changed_rows = sorted({y for t in measured['ledger'] for y in (t['changed_rows'] or [])})
        facts['current_changed_rows'] = [
            {'ref': f'row:{y}', 'row': y,
             'color_counts': {str(c): grid[y].count(c) for c in sorted(set(grid[y]))}}
            for y in changed_rows if y < len(grid)]
        # Exact source excerpt check; no retrospective wording added.
        for _, text, _ in claims:
            if text not in case['actor_claim']: raise ValueError(f'not an exact excerpt: {text}')
        result.append({'id': case['id'], 'narrative': case['actor_claim'], 'facts': facts,
                       'claims': [{'id': id, 'text': text} for id,text,_ in claims],
                       'expected': {id: status for id,_,status in claims},
                       'source': str(path.relative_to(ROOT)), 'cutoff': records[-1]['step'],
                       'prefix_sha256': hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest()})
    return result


def request(prompt, packet):
    body = json.loads((BASELINE / '00.request.json').read_text())
    body['messages'] = [{'role':'system','content':prompt}, {'role':'user','content':json.dumps(packet)}]
    return body


def initial_requests(data):
    requests = []
    for d in data:
        for claim in d['claims']:
            requests.append((f'{d["id"]}-single-{claim["id"]}', request(VERIFY, {'facts':d['facts'],'claims':[claim]})))
    for d in data:
        requests.append((f'{d["id"]}-list', request(VERIFY, {'facts':d['facts'],'claims':d['claims']})))
    for d in data:
        requests.append((f'{d["id"]}-extract', request(EXTRACT, {'narrative':d['narrative']})))
    return requests


def prepare(out):
    data = datasets()
    out.mkdir(parents=True,exist_ok=False)
    (out/'DATA.json').write_text(json.dumps(data,indent=2)+'\n')
    for name,body in initial_requests(data):
        (out/f'{name}.request.json').write_text(json.dumps(body,indent=2)+'\n')
    (out/'PROMPTS.json').write_text(json.dumps({'verify':VERIFY,'extract':EXTRACT},indent=2)+'\n')
    print('Prepared 10 initial requests plus 2 dependent verification calls; no inference.')


def decode_json(content):
    """Accept JSON or one JSON code fence; do not edit the response payload."""
    text = content.strip()
    if text.startswith('```json\n') and text.endswith('\n```'):
        text = text[len('```json\n'):-len('\n```')]
    elif text.startswith('```\n') and text.endswith('\n```'):
        text = text[len('```\n'):-len('\n```')]
    return json.loads(text)


def execute(out, previous=None):
    key = os.environ.get('OPENROUTER_API_KEY')
    for file in [ROOT/'.env',Path('/home/pranav/projects/arc-agi-3/.env')]:
        if key: break
        if file.exists():
            for line in file.read_text().splitlines():
                if line.startswith('OPENROUTER_API_KEY='): key=line.split('=',1)[1].strip().strip('\"\'');break
    if not key: raise ValueError('missing credential')
    card = out/'EXPERIMENT-CARD.md'
    hashes = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.json')}
    with (out/'STARTED.json').open('x') as f:
        json.dump({'time':time.time(),'card_sha256':hashlib.sha256(card.read_bytes()).hexdigest(),
                   'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'input_hashes':hashes,
                   'continued_from':str(previous) if previous else None},f,indent=2)
    result={'status':'completed','calls':0,'cost_usd':0.0,'reserved_usd':0.0,'seconds':0.0}
    reused = {}
    if previous:
        prior = json.loads((previous/'EXECUTION.json').read_text())
        assert prior['status']=='aborted' and prior['calls']==1
        assert json.loads((previous/'DATA.json').read_text())==json.loads((out/'DATA.json').read_text())
        source = next(previous.glob('*.response.json'))
        name = source.name.removesuffix('.response.json')
        assert json.loads((previous/f'{name}.request.json').read_text())==json.loads((out/f'{name}.request.json').read_text())
        raw = source.read_bytes()
        (out/source.name).write_bytes(raw)
        (out/f'{name}.timing.json').write_bytes((previous/f'{name}.timing.json').read_bytes())
        reused[name] = decode_json(json.loads(raw)['choices'][0]['message']['content'])
        for field in ('calls','cost_usd','reserved_usd','seconds'): result[field]=prior[field]
        result['reused_calls']=1
    def call(name,body):
        payload=json.dumps(body).encode()
        reserve=(len(payload)+4096)*.42/1e6+1024*3/1e6
        if result['calls']>=12 or result['reserved_usd']+reserve>.25: raise ValueError('budget exceeded')
        (out/f'{name}.request.json').write_text(json.dumps(body,indent=2)+'\n')
        result['calls']+=1; result['reserved_usd']+=reserve
        req=urllib.request.Request('https://openrouter.ai/api/v1/chat/completions',payload,
                                   {'Content-Type':'application/json','Authorization':'Bearer '+key})
        start=time.monotonic()
        with urllib.request.urlopen(req,timeout=45) as response: raw=response.read()
        seconds=time.monotonic()-start
        (out/f'{name}.response.json').write_bytes(raw)
        parsed=json.loads(raw); choice=parsed['choices'][0];usage=parsed['usage']
        (out/f'{name}.timing.json').write_text(json.dumps({'seconds':seconds,'usage':usage},indent=2))
        result['seconds']+=seconds;result['cost_usd']+=usage['cost']
        assert parsed['provider']=='CoreWeave' and choice['finish_reason']=='stop'
        assert usage['completion_tokens_details']['reasoning_tokens']==0
        assert not choice['message'].get('reasoning') and not choice['message'].get('reasoning_content')
        assert result['cost_usd']<.25
        return decode_json(choice['message']['content'])
    try:
        data=json.loads((out/'DATA.json').read_text())
        replies={}
        for name,_ in initial_requests(data):
            body=json.loads((out/f'{name}.request.json').read_text())
            replies[name]=reused[name] if name in reused else call(name,body)
        for d in data:
            claims=replies[f'{d["id"]}-extract']['claims']
            assert isinstance(claims,list) and 0<len(claims)<=12
            assert len({c['id'] for c in claims})==len(claims)
            assert all(isinstance(c['text'],str) and c['text'] in d['narrative'] for c in claims), 'extraction not exact'
            # No manual additions/repairs: omissions propagate to verification.
            call(f'{d["id"]}-narrative-verify',request(VERIFY,{'facts':d['facts'],'claims':claims}))
    except Exception as e:
        result.update(status='aborted',error=f'{type(e).__name__}: {e}')
        if hasattr(e,'read'): (out/'error-body.txt').write_bytes(e.read())
    (out/'EXECUTION.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--continue-from',type=Path)
    args=parser.parse_args()
    execute(args.output,args.continue_from) if args.execute else prepare(args.output)
