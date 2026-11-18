"""Read-only raw audit adapter; original v3 auditors and evidence stay untouched."""
import hashlib
import importlib.util
import json
import sys
import numpy as np
from reference import errors, paired_summary, qualify_outputs
from pathlib import Path

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'serving-headroom-v3-20261008'
sys.path.insert(0,str(BASE))
spec=importlib.util.spec_from_file_location('qualified_v3_audit',BASE/'audit.py')
v3=importlib.util.module_from_spec(spec);spec.loader.exec_module(v3)


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()



def audit_benchmark(root, cases):
    claimed=json.loads((root/'RESULT.json').read_text())
    completed=claimed['cases_completed']
    if not 1<=completed<=9 or not claimed['sentinel']['passed']:raise ValueError('Incomplete or failed benchmark')
    timings=[]
    for record in cases[:completed]:
        directory=root/Path(record['case']).stem
        result=json.loads((directory/'result.json').read_text())
        if result['record']!=record:raise ValueError('Benchmark case identity differs')
        calls=result['actual_gemm_calls']
        if len(calls)!=4 or [c['use_atomic_add'] for c in calls]!=[True,True,False,False] or not all(c['use_fp32_reduce'] for c in calls):
            raise ValueError('Observed GEMM flags differ')
        numeric=[]
        for variant in ['original','changed','restored']:
            with np.load(directory/f'{variant}-outputs.npz') as arrays:
                numeric.append({'variant':variant,**qualify_outputs(**dict(arrays))})
        if numeric!=result['numerical']:raise ValueError('Numerical summary differs from raw arrays')
        with np.load(directory/'restored-outputs.npz') as graph, np.load(directory/'eager-outputs.npz') as eager:
            agreement=[errors(graph['baseline'],eager['baseline']),errors(graph['treatment'],eager['treatment'])]
        passed=all(r['passed'] for r in numeric) and all(r['finite'] and r['nrms']<=.001 and r['nmax']<=.01 for r in agreement)
        if agreement!=result['eager_graph'] or passed!=result['numerical_passed']:raise ValueError('Eager/graph summary differs')
        if result['inputs_before']!=result['inputs_after']:raise ValueError('Inputs changed')
        if not passed:
            if directory!=root/Path(cases[completed-1]['case']).stem or completed==9 and claimed['advance_to_serving_preparation']:
                raise ValueError('Run continued or advanced after numerical rejection')
            break
        for mode in ['graph','eager']:
            rows=[json.loads(line) for line in (directory/f'{mode}-pairs.jsonl').read_text().splitlines()]
            if len(rows)!=30:raise ValueError('Expected30 paired intervals')
            pairs=[]
            for i,row in enumerate(rows):
                expected=['baseline','treatment'] if i%2==0 else ['treatment','baseline']
                if row['pair']!=i or row['order']!=expected or row['repeats']!=64:raise ValueError('Timing schedule changed')
                pairs.append([row['ms_per_call']['baseline'],row['ms_per_call']['treatment']])
            summary=paired_summary(pairs)
            if summary!=result[f'{mode}_timing']:raise ValueError('Paired timing summary differs')
            if mode=='graph':timings.append(summary)
    eligible=(completed==9 and len(timings)==9 and all(t['geometric_speedup']>=1.03 and t['bootstrap_95'][0]>1.01 and t['faster_pairs']>=27 for t in timings))
    if eligible!=claimed['advance_to_serving_preparation'] or timings!=claimed['case_graph_summaries']:raise ValueError('Advancement verdict differs from raw audit')
    return {'advance_to_serving_preparation':eligible,'cases_completed':completed,'graph_summaries':timings}


def audit(root):
    result={'run_id':HERE.name,'qualified':False}
    try:
        hashes=json.loads((root/'output-hashes.json').read_text())
        if not hashes:raise ValueError('Empty output manifest')
        for name,expected in hashes.items():
            if sha(root/name)!=expected:raise ValueError('Raw hash mismatch: '+name)
        hardware=json.loads((root/'hardware.json').read_text())
        if hardware.get('valid') is not True:raise ValueError('Hardware gate failed')
        state=json.loads((root/'state.json').read_text())
        if state.get('run_id')!=result['run_id'] or state.get('stop') or [b['label'] for b in state['blocks']]!=['PROFILE10']:
            raise ValueError('Incomplete capture state')
        result['capacity']=v3.capacity(json.loads((root/'server-info-before.json').read_text()),json.loads((root/'cohort.json').read_text()))
        contract=json.loads((BASE/'native-contract.json').read_text())
        if json.loads((root/'package-identity.json').read_text())!=contract['wheel']:raise ValueError('Wheel identity mismatch')
        native=json.loads((root/'runtime-source/identity.json').read_text())['files']
        for name,expected in contract['files'].items():
            if native.get(name)!=expected or sha(root/'runtime-source'/name)!=expected:raise ValueError('Native source mismatch: '+name)
        if json.loads((root/'patch-manifest.json').read_text())!=json.loads((HERE/'patch-manifest.json').read_text()):
            raise ValueError('Candidate patch manifest mismatch')
        result['blocks']=[v3.verify_block(root,'WARMUP',10),v3.verify_block(root,'PROFILE10',10)]
        requests=sorted((root/'PROFILE10').glob('*-request.json'))
        for path in requests:
            original=BASE/'readout/raw/headroom/PROFILE10'/path.name
            if json.loads(path.read_text())!=json.loads(original.read_text()):raise ValueError('Original fixed wire cohort differs: '+path.name)
        capture=root/'capture'
        binding=json.loads((capture/'binding-target.json').read_text())
        draft=json.loads((capture/'binding-draft.json').read_text())
        if binding['is_draft_worker'] or len(binding['modules'])!=48 or draft['modules'] or draft['is_draft_worker'] is not True:
            raise ValueError('Target-only binding differs')
        if set(binding['selected'])!={'layer-00','layer-24','layer-47'}:raise ValueError('Selected layer set mismatch')
        cases=[json.loads(line) for line in (capture/'cases.jsonl').read_text().splitlines()]
        expected={f'layer-{layer:02d}-replay-{step:02d}.pt' for layer in [0,24,47] for step in [8,16,32]}
        if len(cases)!=9 or {r['case'] for r in cases}!=expected:raise ValueError('Expected nine fixed captured cases')
        for row in cases:
            if row['raw_bs']!=10 or row['mode']!='TARGET_VERIFY' or row['rows']!=row['hidden_shape'][0]:raise ValueError('Unsupported replay layout')
            for name,field in [('case','sha256'),('weights','weights_sha256')]:
                if sha(capture/row[name])!=row[field]:raise ValueError('Captured tensor hash mismatch')
        result['verified_raw_hashes']=len(hashes);result['qualified']=True
        result['benchmark']=audit_benchmark(root/'benchmark',cases)
        result['advance_to_serving_preparation']=result['benchmark']['advance_to_serving_preparation']
    except (OSError,ValueError,RuntimeError,KeyError,TypeError,EOFError,ZeroDivisionError) as error:
        result['qualified']=False;result['error']=repr(error)
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('--output',type=Path)
    args=parser.parse_args();result=audit(args.root)
    if args.output:args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));raise SystemExit(0 if result['qualified'] else 1)
