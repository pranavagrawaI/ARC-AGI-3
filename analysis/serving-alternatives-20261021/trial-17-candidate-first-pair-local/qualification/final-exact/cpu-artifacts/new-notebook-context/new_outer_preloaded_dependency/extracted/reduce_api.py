"""Unchanged within-arm API gates with explicit reversed pairing."""
import argparse,json
from pathlib import Path
import accounting,catalog,load_contract as contract,provenance
from api_arm import arm

def compare(root):
    root=Path(root);cat=catalog.load();a=arm(root/'control',cat,'control');b=arm(root/'candidate',cat,'candidate')
    issues=[];mix=None;ratio=None;binding=None
    if not a['valid'] or not b['valid']:issues.append('invalid arm')
    else:
        try:binding=provenance.check(root,a,b)
        except (ValueError,KeyError,TypeError,OSError,StopIteration) as error:issues.append(str(error) or type(error).__name__)
        for k in ('session_id','gpu_uuid','boot_id'):
            if a['capture'][k]!=b['capture'][k]:issues.append('pair capture identity '+k)
        try:
            import handoff
            handoff.pair(b['capture'],a['capture'])
        except (ValueError,KeyError,TypeError):issues.append('fresh candidate-first sequential loads')
        mix=contract.mix(a['measured_rows'],b['measured_rows'])
        if any(mix[k]>.10 for k in ('prompt_tvd','images_tvd','cap_tvd')):issues.append('reached workload mix')
        if mix['early_stop_difference']>.05 or not .9<=mix['completion_fraction_ratio']<=1.1:issues.append('output-length mix')
        if not issues:ratio=b['U_resp']/a['U_resp']
    for value in (a,b):value.pop('measured_rows',None)
    verdict='FEASIBILITY_ONLY' if issues else 'CLOSE_TESTED_CONFIG' if ratio<1.20 else 'QUALITY_INVESTIGATION_ONLY'
    return dict(estimand='WIRE_EQUIVALENT_FIXED_CANDIDATE_FIRST_TOTAL_PACKAGE_API_CAPACITY',verdict=verdict,issues=issues,ratio=ratio,mix=mix,control=a,candidate=b,package_binding=binding,
                limitations=accounting.LIMITATIONS+['One fixed-order pair assumes temporal stability; does not identify an engine-only effect.','Full-reply quality and production/adoption remain unqualified.'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--arm',choices=['control','candidate']);a=p.parse_args()
    value=arm(Path(a.root)/a.arm,catalog.load(),a.arm) if a.arm else compare(a.root)
    value.pop('measured_rows',None);print(json.dumps(value,indent=2))
    Path(a.root,('result-'+a.arm if a.arm else 'paired-result')+'.json').write_text(json.dumps(value,indent=2)+'\n')
    raise SystemExit(0 if value.get('valid',True) else 1)
