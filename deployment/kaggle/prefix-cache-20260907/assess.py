"""Describe matched effects; never automatically promote a configuration."""
import json
from pathlib import Path
import statistics
import sys


def assess(root):
    root=Path(root)
    arms=[]
    for index,chunk in enumerate([2048,4096,4096,2048]):
        path=root/f'arm-{index}-chunk-{chunk}'
        lifecycle=json.loads((path/'lifecycle.json').read_text())
        if lifecycle.get('status')!='passed':raise ValueError('Unqualified lifecycle')
        conditions={p.stem:json.loads(p.read_text()) for p in path.glob('*-condition.json')}
        batches={p.stem:json.loads(p.read_text()) for p in path.glob('batch-*.json')}
        if len(conditions)!=6 or len(batches)!=4:raise ValueError('Incomplete arm')
        natural=json.loads((path/'natural-results.json').read_text())
        if len(natural)!=9:raise ValueError('Incomplete natural validation')
        arms.append({'index':index,'chunk':chunk,'conditions':conditions,'batches':batches,'natural':natural,
                     'resolved_policy':json.loads((path/'resolved-policy.json').read_text())})
    conditions=[]
    for name in sorted(arms[0]['conditions']):
        rows=[a['conditions'][name] for a in arms]
        successors=[r['successor'] for r in rows]
        cached=[r['cached_input_tokens'] for r in successors]
        ids=[]
        for a,r in zip(arms,rows):
            request=root/f"arm-{a['index']}-chunk-{a['chunk']}"/(r['successor']['name']+'-request.json')
            data=json.loads(request.read_text())
            ids.append(data['input_ids'])
            if data['sampling_params']!={'temperature':1.0,'top_k':20,'top_p':.95,'max_new_tokens':256,'ignore_eos':True}:
                raise ValueError('Sampling/count drift')
        if any(x!=ids[0] for x in ids):raise ValueError('Successor input drift')
        fingerprints=[r.get('output_fingerprint') for r in successors]
        conditions.append({'condition':name,'cached_tokens_abba':cached,
            'cache_counts_match':len(set(cached))==1,
            'successor_e2e_s_abba':[r['client_e2e_s'] for r in successors],
            'prefill_s_abba':[r['server_forward_to_prefill_finished_s'] for r in successors],
            'condition_wall_s_abba':[r['condition_wall_s'] for r in rows],
            'output_fingerprints_abba':fingerprints,
            'output_equality':'unknown' if any(not f or all(v is None for v in f.values()) for f in fingerprints) else ('matched' if all(f==fingerprints[0] for f in fingerprints) else 'diverged'),
            'interpretation':'Cache-count changes are part of chunk policy; equal counts do not isolate GPU kernel effects'})
    batches=[]
    for name in sorted(arms[0]['batches']):
        rows=[a['batches'][name] for a in arms]
        if any(r['input_sha256']!=rows[0]['input_sha256'] for r in rows):raise ValueError('Batch input drift')
        batches.append({'workload':name,'wall_s_abba':[r['wall_s'] for r in rows],
            'median_ready_s_abba':[statistics.median(q['ready_to_completed_s'] for q in r['requests']) for r in rows],
            'max_ready_s_abba':[max(q['ready_to_completed_s'] for q in r['requests']) for r in rows],
            'within180_abba':[r['completed_within_180s_from_ready'] for r in rows],
            'cached_tokens_abba':[{q['name']:q['cached_input_tokens'] for q in r['requests']} for r in rows],
            'server_overlap':'Requires metrics windows; client concurrency alone does not qualify overlap'})
    natural=[{'arm_index':a['index'],'chunk':a['chunk'],'requests':[
        {k:r.get(k) for k in ('name','client_e2e_s','usage','quality','cached_input_tokens','new_input_tokens')}
        for r in a['natural']]} for a in arms]
    return {'status':'MEASURED_REVIEW_REQUIRED','promotion':False,'conditions':conditions,'batches':batches,'natural':natural,
            'resolved_policies':[a['resolved_policy'] for a in arms],
            'limits':'Two engine runs per configuration; mechanical95k and same-game prefix bias; no actual summary boundary or game deadlines. Stochastic output fingerprints are descriptive, not an equivalence or quality test.'}

if __name__=='__main__':
    root=Path(sys.argv[1]);result=assess(root)
    (root/'COMPARISON.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Comparison saved; promotion=false')
