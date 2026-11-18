#!/usr/bin/env python3
"""Deterministic local gate. Missing final review is prepared, never approved."""
import hashlib, json
from pathlib import Path
HERE=Path(__file__).resolve().parent; REPO=HERE.parents[1]; PACKAGE=REPO/'deployment/kaggle'/HERE.name
R4A=REPO/'analysis/construction-policy-integration-20260913T221412Z-r4'; R4P=REPO/'deployment/kaggle/construction-policy-integration-20260913T221412Z-r4'
R4={'manifest':'577c8d8b326d826c7e48e5b9a4e5cb668005eb4bd52c06e6d4ed4ed131fce451','notebook':'e4b9f30f8fbd84eefa3b1b84b2a00f19a379d9ee381b4a5e60d4141e794bfa35','build':'c4beed81a87f4ba667aedd0364c8132d7ea1c168ee7ad7c1cc06249f25df8f27'}
SOURCE=['PROVIDER-SELECTION.json','accelerator.py','case.json','constructor.py','containment_base.py','inspection_runner.py','integration.py','integration_boundary.py','local_transport.py','model-metadata.json','qualified_runner.py','real_trace.py','request_contract.py']
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p): return json.loads(p.read_text())
def review_valid(review,manifest_sha,card_sha):
    return review=={'status':'RESOLVED','identity':HERE.name,'manifest_sha256':manifest_sha,'card_sha256':card_sha,'notebook_sha256':R4['notebook']}
def authorization_valid(auth,manifest_sha,card_sha):
    return isinstance(auth,dict) and bool(auth) and auth.get('used') is False and auth.get('identity')==HERE.name and auth.get('manifest_sha256')==manifest_sha and auth.get('card_sha256')==card_sha and auth.get('maximum_pushes')==1 and auth.get('retries')==0
def check(write=True):
    errors=[]
    if sha(R4A/'MANIFEST.json')!=R4['manifest']: errors.append('R4_MANIFEST_DRIFT')
    if sha(R4P/'construction.ipynb')!=R4['notebook'] or sha(R4P/'BUILD.json')!=R4['build']: errors.append('R4_PACKAGE_DRIFT')
    prereq=load(HERE/'R4-REVIEW-PREREQUISITE.json')
    if prereq.get('status')!='RESOLVED' or prereq.get('manifest_sha256')!=R4['manifest'] or prereq.get('verification')!={'integration':21,'constructor':46,'containment':53}: errors.append('R4_REVIEW_PREREQUISITE')
    for n in SOURCE:
        if sha(HERE/n)!=sha(R4A/n): errors.append('R4_SOURCE_DRIFT:'+n)
    if sha(PACKAGE/'remote_entry.py')!=sha(R4P/'remote_entry.py'): errors.append('R4_SOURCE_DRIFT:remote_entry.py')
    if sha(PACKAGE/'construction.ipynb')!=R4['notebook']: errors.append('NOTEBOOK_BYTES_DRIFT')
    expected=load(HERE/'SERVER-REFERENCE.json'); meta=load(PACKAGE/'kernel-metadata.json')
    if meta!=expected: errors.append('SERVER_REFERENCE_DRIFT')
    if expected['competition_sources']!=['arc-prize-2026-arc-agi-3'] or expected['dataset_sources']!=load(REPO/'deployment/kaggle/submission-20260912-v15/kernel-metadata.json')['dataset_sources']: errors.append('SOURCE_CONTRACT_DRIFT')
    if not (expected['is_private'] is True and expected['enable_internet'] is False and expected['machine_shape']=='NvidiaRtxPro6000' and '@sha256:' in expected['docker_image']): errors.append('REMOTE_POLICY_DRIFT')
    manifest=load(HERE/'MANIFEST.json'); manifest_sha=sha(HERE/'MANIFEST.json')
    for rel,want in manifest['files'].items():
        path=(HERE/rel.removeprefix('analysis/')) if rel.startswith('analysis/') else (PACKAGE/rel.removeprefix('package/'))
        if not path.is_file() or sha(path)!=want: errors.append('MANIFEST_DRIFT:'+rel)
    auth_path=HERE/'AUTHORIZATION.json'; consumed=HERE/'LAUNCH-CONSUMED.json'
    if not auth_path.is_file(): errors.append('AUTHORIZATION_MISSING'); auth={}
    else: auth=load(auth_path)
    if not authorization_valid(auth,manifest_sha,sha(HERE/'EXPERIMENT-CARD.md')): errors.append('AUTHORIZATION_BINDING')
    if consumed.exists(): errors.append('AUTHORIZATION_CONSUMED')
    review_path=HERE/'REVIEW-RESOLVED.json'
    if not review_path.exists(): approval='MISSING'
    else: approval='RESOLVED' if review_valid(load(review_path),manifest_sha,sha(HERE/'EXPERIMENT-CARD.md')) else 'REJECTED_OR_MISMATCHED'
    if approval!='RESOLVED': status='PREPARED_NOT_APPROVED' if not errors else 'HOLD'
    else: status='APPROVED_NOT_LAUNCHED' if not errors else 'HOLD'
    out={'status':status,'approved':status=='APPROVED_NOT_LAUNCHED','errors':errors,'final_review':approval,'identity':HERE.name,'manifest_sha256':manifest_sha,'card_sha256':sha(HERE/'EXPERIMENT-CARD.md'),'notebook_sha256':sha(PACKAGE/'construction.ipynb'),'authorization_present':auth_path.exists(),'authorization_unconsumed':not consumed.exists(),'r4_hash_inheritance':not any(x.startswith('R4_') or x.startswith('NOTEBOOK') for x in errors),'remote_containment_qualified':'16/16 inherited exact qualified_runner hash '+sha(HERE/'qualified_runner.py')}
    if write:
        (HERE/'evidence').mkdir(exist_ok=True); (HERE/'evidence/PRELAUNCH.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    return out
if __name__=='__main__':
    result=check(); print(json.dumps(result,indent=2,sort_keys=True)); raise SystemExit(0 if result['status'] in {'PREPARED_NOT_APPROVED','APPROVED_NOT_LAUNCHED'} else 2)
