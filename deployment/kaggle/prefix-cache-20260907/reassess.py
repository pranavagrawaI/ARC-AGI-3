"""Local successor assessment. Never edit the launched notebook or raw run files."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import statistics

CHUNKS = (2048, 4096, 4096, 2048)
BATCH_NAMES = tuple(f'batch-{workload}-{c}' for workload in ('captured', 'long95000') for c in (1, 4))
WAIT_LINE = re.compile(r'^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] INFO: +Waiting for background tasks to complete\. \(CTRL\+C to force quit\)\r?\n?$')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lifecycle_review(path):
    original = read(path / 'lifecycle.json')
    classification = read(path / 'teardown-log-classification.json')
    log = (path / 'server.log').read_bytes()
    revised = copy.deepcopy(classification)
    changes = []
    offset = 0
    source_end = None
    for entry in revised['entries']:
        start, end = entry['source_byte_start'], entry['source_byte_end']
        if log[start:end].decode() != entry['text']:
            raise ValueError('Classified text differs from retained server bytes')
        if entry['byte_start'] != offset or entry['byte_end']-offset != end-start:
            raise ValueError('Classification byte coverage mismatch')
        if source_end is not None and source_end != start:
            raise ValueError('Classification source gap')
        source_end = end
        offset = entry['byte_end']
        if not entry['allowed'] and entry['class'] == 'forbidden_unclassified_log_line' and entry['rule'] == 'default_deny_unmatched_nonempty_line' and WAIT_LINE.fullmatch(entry['text']):
            changes.append(copy.deepcopy(entry))
            entry.update(allowed=True, **{'class':'allowed_normal_event'}, rule='uvicorn_wait_background_tasks_exact')
    if offset != classification['bytes']:
        raise ValueError('Incomplete classified byte coverage')
    revised['forbidden_count'] = sum(not e['allowed'] for e in revised['entries'])
    revised['allowed'] = revised['forbidden_count'] == 0
    gates = (original.get('containment_passed') is True,
             original.get('quiet_window_valid') is True,
             original.get('term_only') is True,
             type(original.get('root_returncode')) is int and original['root_returncode'] == 0)
    permitted_failure = original.get('status') == 'passed' or (
        original.get('status') == 'failed' and original.get('failure_classification') == 'teardown_log_unacceptable'
        and original.get('log_classification_allowed') is False and bool(changes))
    allowed = all(gates) and permitted_failure and revised['allowed']
    return {'original_lifecycle':original, 'reassessed_status':'passed' if allowed else 'failed',
            'changed_entries':changes, 'revised_classification':revised,
            'rule':'Exact INFO background-task-wait message only; all other denials and lifecycle gates retained',
            'evidence_sha256':{n:sha(path/n) for n in ('lifecycle.json','server.log','teardown-log-classification.json')}}


def batch_summaries(path):
    # The directory also contains requests/results/events. Those are evidence,
    # not aggregate workload summaries, and must never enter this mapping.
    return {name:read(path/(name+'.json')) for name in BATCH_NAMES}


def assess(root):
    arms=[]
    for index,chunk in enumerate(CHUNKS):
        path=root/f'arm-{index}-chunk-{chunk}'
        lifecycle=lifecycle_review(path)
        if lifecycle['reassessed_status']!='passed':
            raise ValueError(f'Unqualified lifecycle after narrow review: arm {index}')
        conditions={p.stem:read(p) for p in path.glob('*-condition.json')}
        if len(conditions)!=6:raise ValueError('Incomplete conditions')
        natural=read(path/'natural-results.json')
        if len(natural)!=9:raise ValueError('Incomplete natural validation')
        arms.append({'path':path,'index':index,'chunk':chunk,'lifecycle':lifecycle,
                     'conditions':conditions,'batches':batch_summaries(path),'natural':natural,
                     'policy':read(path/'resolved-policy.json')})
    comparisons=[]
    for name in sorted(arms[0]['conditions']):
        rows=[a['conditions'][name] for a in arms]
        successors=[r['successor'] for r in rows]
        bodies=[read(a['path']/(r['successor']['name']+'-request.json')) for a,r in zip(arms,rows)]
        if any(b!=bodies[0] for b in bodies):raise ValueError('Successor input/policy drift')
        expected={'temperature':1.0,'top_k':20,'top_p':.95,'max_new_tokens':256,'ignore_eos':True}
        if bodies[0]['sampling_params']!=expected:raise ValueError('Sampling/count drift')
        if any(r['meta_info']['completion_tokens']!=256 for r in successors):raise ValueError('Actual output count drift')
        comparisons.append({'condition':name,'cached_tokens_abba':[r['cached_input_tokens'] for r in successors],
            'successor_e2e_s_abba':[r['client_e2e_s'] for r in successors],
            'prefill_s_abba':[r['server_forward_to_prefill_finished_s'] for r in successors],
            'condition_wall_s_abba':[r['condition_wall_s'] for r in rows]})
    batches=[]
    for name in BATCH_NAMES:
        rows=[a['batches'][name] for a in arms]
        if any(r['input_sha256']!=rows[0]['input_sha256'] for r in rows):raise ValueError('Batch input drift')
        if any(len(r['requests'])!=4 or any(q['meta_info']['completion_tokens']!=256 for q in r['requests']) for r in rows):
            raise ValueError('Incomplete batch output')
        batches.append({'workload':name,'wall_s_abba':[r['wall_s'] for r in rows],
            'median_ready_s_abba':[statistics.median(q['ready_to_completed_s'] for q in r['requests']) for r in rows],
            'max_ready_s_abba':[max(q['ready_to_completed_s'] for q in r['requests']) for r in rows],
            'within180_abba':[r['completed_within_180s_from_ready'] for r in rows]})
    return {'status':'LOCAL_REASSESSMENT_COMPLETE','original_kaggle_status':'ERROR','promotion':False,
            'selection':'retain chunk 2048; candidate did not meet 10 percent improvement gate',
            'lifecycle':[a['lifecycle'] for a in arms],'conditions':comparisons,'batches':batches,
            'resolved_policies':[a['policy'] for a in arms],
            'natural':[{ 'arm':a['index'],'requests':a['natural']} for a in arms],
            'limits':'No new execution; exact one-line log classification amendment. Same-game, mechanical95k, no summary boundary or game-deadline qualification.'}


def snapshot(root):
    return {str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    root=args.root.resolve();out=args.out.resolve()
    if out==root or root in out.parents:raise ValueError('Output must be outside raw evidence')
    if out.exists():raise ValueError('Use a new output directory; no overwrite')
    before=snapshot(root)
    result=assess(root)
    after=snapshot(root)
    if before!=after:raise ValueError('Raw evidence changed during assessment')
    out.mkdir(parents=True)
    result['raw_files_sha256']=before
    result['assessor_sha256']=sha(Path(__file__))
    (out/'REASSESSMENT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'output':str(out),'raw_files_unchanged':len(before),'status':result['status'],'promotion':False}))


if __name__=='__main__':main()
