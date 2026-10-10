"""Recompute diagnostic claims from collected arrays/pairs; no serving gate."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from benchmark import repeatability_metrics
from reference import errors, paired_summary, qualify_outputs

HERE=Path(__file__).resolve().parent


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def timing(case,mode):
    rows=[json.loads(l) for l in (case/f'{mode}-pairs.jsonl').read_text().splitlines()]
    if len(rows)!=30:raise ValueError('Missing timing pairs')
    pairs=[]
    for i,row in enumerate(rows):
        order=['baseline','treatment'] if i%2==0 else ['treatment','baseline']
        if row['pair']!=i or row['order']!=order or row['repeats']!=64:
            raise ValueError('Timing pairing/order differs')
        pairs.append([row['ms_per_call']['baseline'],row['ms_per_call']['treatment']])
    return paired_summary(pairs)


def audit_case(case,record):
    result=json.loads((case/'result.json').read_text())
    if result['record']!=record:raise ValueError('Captured case identity differs')
    calls=result['actual_gemm_calls']
    if len(calls)!=4 or [c['use_atomic_add'] for c in calls]!=[True,True,False,False] or not all(c['use_fp32_reduce'] for c in calls):
        raise ValueError('Actual GEMM contrast differs')
    if result['inputs_before']!=result['inputs_after']:raise ValueError('Input restoration failed')
    summaries=[]
    for claimed,variant in zip(result['numerical'],['original','changed','restored'],strict=True):
        arrays=dict(np.load(case/f'{variant}-outputs.npz'))
        if not all(np.isfinite(v).all() for v in arrays.values()):raise ValueError('Nonfinite saved output/reference')
        computed=qualify_outputs(**arrays)
        poison=computed['poison_agreement']
        if not poison['finite'] or poison['nrms']>.001 or poison['nmax']>.01:
            raise ValueError('Scratch poison disagreement')
        for key,value in computed.items():
            if claimed[key]!=value:raise ValueError('Numerical claim differs from saved arrays')
        repeated=dict(np.load(case/f'{variant}-repeats.npz'))
        if set(repeated)!={'baseline','treatment','independent_control'} or any(v.shape!=(8,*arrays['baseline'].shape) for v in repeated.values()):
            raise ValueError('Repetition count/shape differs')
        if not all(np.isfinite(v).all() for v in repeated.values()):raise ValueError('Nonfinite repetition')
        variability=repeatability_metrics(**repeated,fp32=arrays['fp32'])
        if variability!=claimed['repeatability']:raise ValueError('Repeatability claim differs from saved arrays')
        dtype=np.load(case/f'{variant}-dtype-reference.npz')['dtype_reference']
        if not np.isfinite(dtype).all():raise ValueError('Nonfinite dtype reference')
        dtype_metrics={arm:errors(arrays[arm],dtype) for arm in ['baseline','treatment']}
        if dtype_metrics!=claimed['dtype_reference']:raise ValueError('Dtype diagnostic differs')
        summaries.append({'variant':variant,'v3_numerical_passed':computed['passed'],
                          'control_max_abs_envelope':variability['control_max_abs_envelope'],
                          'cross_arm_max_abs':max(x['max_abs'] for x in variability['paired_treatment_control']),
                          'max_cross_arm_exceeds_control_envelope':variability['max_cross_arm_exceeds_control_envelope'],
                          'dtype_reference':dtype_metrics})
    passed=all(row['v3_numerical_passed'] for row in summaries)
    eager=dict(np.load(case/'eager-outputs.npz'));restored=dict(np.load(case/'restored-outputs.npz'))
    agreement=[errors(restored[arm],eager[arm]) for arm in ['baseline','treatment']]
    if agreement!=result['eager_graph'] or not all(m['finite'] for m in agreement):
        raise ValueError('Eager/graph evidence differs or is nonfinite')
    passed=passed and all(m['nrms']<=.001 and m['nmax']<=.01 for m in agreement)
    if result['numerical_passed']!=passed:raise ValueError('Original gate claim differs')
    timings={mode:timing(case,mode) for mode in ['graph','eager','control-graph']}
    for mode,key in [('graph','graph_timing'),('eager','eager_timing'),('control-graph','control_graph_timing')]:
        if timings[mode]!=result[key]:raise ValueError('Timing claim differs from raw pairs')
    return {'case':case.name,'numerical':summaries,'timings':timings,'eager_graph':agreement,'v3_numerical_passed':passed}


def audit(raw):
    report={'qualified':False,'advance_to_serving_preparation':False}
    try:
        hashes=json.loads((raw/'output-hashes.json').read_text())
        if not hashes:raise ValueError('No output hashes')
        actual={str(p.relative_to(raw)) for p in raw.rglob('*') if p.is_file() and p.name!='output-hashes.json'}
        if set(hashes)!=actual:raise ValueError('Output hash manifest is incomplete')
        for name,digest in hashes.items():
            if sha(raw/name)!=digest:raise ValueError('Raw output hash mismatch: '+name)
        source=json.loads((HERE/'capture-source.json').read_text())
        mounted=json.loads((raw/'capture-source.json').read_text())
        if mounted['kernel_id']!=source['kernel_id'] or mounted['version']!=source['version'] or mounted['files']!=source['files']:
            raise ValueError('Mounted source differs')
        package=json.loads((raw/'package-identity.json').read_text())
        if package['sha256']!=source['source_identity']['wheel_sha256']:raise ValueError('Wheel identity differs')
        installed=json.loads((raw/'installed-patch.json').read_text())
        expected=json.loads((HERE.parent/'marlin-reduction-replay-v3-20261010/patch-manifest.json').read_text())
        if installed!=expected:raise ValueError('Installed candidate differs')
        if not json.loads((raw/'hardware.json').read_text())['valid']:raise ValueError('Wrong hardware')
        top=json.loads((raw/'benchmark/RESULT.json').read_text())
        if top['cases_completed']!=9 or not top['sentinel']['passed'] or top['advance_to_serving_preparation']:
            raise ValueError('Incomplete diagnostic or inappropriate advancement')
        cases=[audit_case(raw/'benchmark'/Path(r['case']).stem,r) for r in source['records']]
        null_valid=all(.97<=c['timings']['control-graph']['geometric_speedup']<=1.03 for c in cases)
        gain=all(c['timings']['graph']['geometric_speedup']>=1.03 and c['timings']['graph']['bootstrap_95'][0]>1.01 and c['timings']['graph']['faster_pairs']>=27 for c in cases)
        report.update(qualified=True,verified_raw_hashes=len(hashes),cases=cases,
                      control_timing_null_valid=null_valid,practical_speed_candidate=bool(gain and null_valid),
                      all_cases_pass_v3_numerical_gate=all(c['v3_numerical_passed'] for c in cases),
                      limits=['one GPU session','diagnostic BF16 model is not an acceptance oracle','no serving or gameplay qualification'])
    except Exception as e:report['error']=str(e)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('raw',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=audit(args.raw)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))
    raise SystemExit(0 if result['qualified'] else 1)
