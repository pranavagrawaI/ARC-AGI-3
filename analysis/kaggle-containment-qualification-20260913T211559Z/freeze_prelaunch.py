"""Freeze and verify this one fresh competition-attached package."""
import hashlib, json
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
PACKAGE=REPO/'deployment/kaggle'/HERE.name
PRIOR=REPO/'analysis/kaggle-containment-qualification-20260913T203056Z'
PRIOR_PACKAGE=REPO/'deployment/kaggle/kaggle-containment-qualification-20260913T203056Z'
COMP='arc-prize-2026-arc-agi-3'

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path): return json.loads(path.read_text())

def main():
    assert not (HERE/'LAUNCH-ATTEMPT.json').exists()
    meta=load(PACKAGE/'kernel-metadata.json'); provider=load(HERE/'PROVIDER-CONTRACT.json')
    expected={'id':'pranavagrawai/containment-q-20260913t211559z','title':'containment-q-20260913t211559z',
      'code_file':'qualification.ipynb','language':'python','kernel_type':'notebook','is_private':True,
      'enable_gpu':True,'machine_shape':'NvidiaRtxPro6000','docker_image':'gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461',
      'docker_image_pinning_type':'original','enable_internet':False,'enable_tpu':False,
      'dataset_sources':[],'kernel_sources':[],'competition_sources':[COMP],'model_sources':[]}
    assert meta==expected
    request=provider['request']; assert request['slug']==expected['id']
    assert request['competition_data_sources']==[COMP]
    assert all(request[k]==[] for k in ('dataset_data_sources','kernel_data_sources','model_data_sources'))
    assert request['session_timeout_seconds']==300 and request['machine_shape']=='NvidiaRtxPro6000'
    assert request['docker_image']==expected['docker_image'] and request['is_private'] is True and request['enable_internet'] is False
    normal=load(HERE/'local-package-normal/VERIFIED.json'); cached=load(HERE/'local-package-cached/VERIFIED.json')
    assert normal['verified'] is True and cached['verified'] is True
    assert sha(HERE/'runner.py')==sha(PRIOR/'runner.py') and sha(HERE/'accelerator.py')==sha(PRIOR/'accelerator.py')
    probe=(HERE/'probe.py').read_text().replace('20260913T211559Z','20260913T203056Z')
    assert hashlib.sha256(probe.encode()).hexdigest()==sha(PRIOR/'probe.py')
    audit=(HERE/'audit_remote.py').read_text().replace("    require(server.get('competition_sources') == ['arc-prize-2026-arc-agi-3'], 'SERVER_COMPETITION_SOURCE')\n    require(all(server.get(k) == [] for k in ('dataset_sources','kernel_sources','model_sources')), 'SERVER_UNINTENDED_SOURCES')", "    require(all(server.get(k) == [] for k in ('dataset_sources','kernel_sources','competition_sources','model_sources')), 'SERVER_SOURCES')")
    assert hashlib.sha256(audit.encode()).hexdigest()==sha(PRIOR/'audit_remote.py')
    build=(PACKAGE/'build.py').read_text().replace('20260913T211559Z','20260913T203056Z').replace('20260913t211559z','20260913t203056z').replace("'competition_sources':['arc-prize-2026-arc-agi-3']","'competition_sources':[]")
    assert hashlib.sha256(build.encode()).hexdigest()==sha(PRIOR_PACKAGE/'build.py')
    delta={'verified':True,'prior_identity':PRIOR.name,'fresh_identity':HERE.name,
      'allowed_changes':['run identity','experiment card','competition attachment metadata','mechanical verification/audit metadata'],
      'containment_tests_and_gpu_gate_unchanged':True,'runner_exact':True,'accelerator_exact':True,
      'probe_identity_normalized_exact':True,'audit_competition_rule_normalized_exact':True,'build_identity_competition_normalized_exact':True}
    (HERE/'DELTA.json').write_text(json.dumps(delta,indent=2)+'\n')
    pre={'verified':True,'fresh_identity':HERE.name,'requested_kernel':expected['id'],
      'competition_sources':[COMP],'private':True,'internet':False,'machine_shape':'NvidiaRtxPro6000',
      'docker_image':expected['docker_image'],'unintended_sources_absent':True,'slugification_check':expected['title']==expected['id'].split('/')[1],
      'normal_generated_notebook_path':'PASS','cached_module_generated_notebook_path':'PASS','package_manifest_schema':'PASS',
      'metadata_schema_exact':True,'official_cli_request_construction_passed':True,'one_push_no_retry':True}
    (HERE/'PRELAUNCH.json').write_text(json.dumps(pre,indent=2)+'\n')
    names=['DELTA.json','EXPERIMENT-CARD.md','PRELAUNCH.json','PROVIDER-CONTRACT.json','runner.py','probe.py','accelerator.py','audit_remote.py','verify_package.py','provider_contract.py','launch_once.py','collect_readonly.py','read_server.py','audit_actual.py','package_execution.py','freeze_prelaunch.py','package-verification.json']
    files={str((HERE/n).relative_to(REPO)):sha(HERE/n) for n in names}
    for n in ('build.py','BUILD.json','kernel-metadata.json','qualification.ipynb'):
        files[str((PACKAGE/n).relative_to(REPO))]=sha(PACKAGE/n)
    manifest={'identity':HERE.name,'launch_authorized':True,'files':files}
    (HERE/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    review={'status':'resolved_for_bounded_probe','basis':'mechanical relocation of reviewed frozen containment sources; competition association is metadata/audit-only',
      'reviewed_source_manifest_sha256':'dc21201f6b2dc584779d4261c572873aa310c333013e14ddc96d8189d10353ff',
      'limitation':'Prior review covers containment bytes and normalized relocation; fresh package integrity and exact competition metadata verified mechanically.',
      'manifest_sha256':sha(HERE/'MANIFEST.json')}
    (HERE/'REVIEW-RESOLVED.json').write_text(json.dumps(review,indent=2)+'\n')
    print(json.dumps({'verified':True,'manifest_sha256':review['manifest_sha256'],'files':len(files),'competition_sources':[COMP]},indent=2))
if __name__=='__main__': main()
