"""Offline post-run audit of sole admission, including explicit deviations.
Never modifies frozen runner, notebook, card, launch or remote system.
"""
import csv,hashlib,json,re
from pathlib import Path
from decimal import Decimal
import audit_remote
from launch_once import reconcile,HERE,PACKAGE

def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 reconcile()
 build=load(PACKAGE/'BUILD.json');launch=load(HERE/'launch.json');assert launch['returncode']==0
 m=re.search(r'Kernel version (\d+) successfully pushed\.  Please check progress at https://www.kaggle.com/code/(\S+)',launch['stdout']);assert m
 version=int(m[1]);kernel=m[2];assert version==1
 assert launch['arguments']==['kernels','push','--path',str(PACKAGE),'--timeout','300','--accelerator','NvidiaRtxPro6000']
 status=load(HERE/'status-exact-version-final.json');server=load(HERE/'SERVER.json');sr=load(HERE/'SERVER-RECEIPT.json')
 assert sr['returncode']==0 and sr['returned_ref']==kernel and sr['returned_version']==version
 assert sr['source_sha256']==sha(HERE/'server-notebook.ipynb')
 # Kaggle reformats JSON. Require complete parsed notebook equality, preserving
 # exact code strings and every embedded byte; never normalize executable code.
 local_nb=load(PACKAGE/'qualification.ipynb');server_nb=load(HERE/'server-notebook.ipynb');assert local_nb==server_nb
 canonical=json.dumps(server_nb).encode();wire=load(HERE/'WIRE-CONTRACT.json')
 assert wire['local_notebook_sha256']==sha(PACKAGE/'qualification.ipynb')
 assert hashlib.sha256(canonical).hexdigest()==wire['wire_notebook_sha256']
 contract={**build,'kernel':kernel,'version':version,'image':server['docker_image'],'notebook_sha256':wire['wire_notebook_sha256']}
 assert contract['image']==load(PACKAGE/'kernel-metadata.json')['docker_image']
 # Preserve strict raw-contract rejection, separate from launch-bound audit.
 strict={**contract,'kernel':build['kernel'],'notebook_sha256':build['files']['qualification.ipynb']}
 try:audit_remote.audit_remote(HERE/'remote/qualification',strict,status,server,(HERE/'server-notebook.ipynb').read_bytes())
 except ValueError as e:strict_rejection=str(e)
 else:raise AssertionError('Expected explicit slug discrepancy to reject strict prelaunch target')
 report=audit_remote.audit_remote(HERE/'remote/qualification',contract,status,server,canonical)
 retrieval=load(HERE/'RETRIEVAL.json');assert retrieval['kernel']==kernel and retrieval['version']==version
 expected={'qualification/result.json','qualification/platform.json','package-lifecycle.json','package-worker.json'}
 assert {x['path'] for x in retrieval['retained']}==expected
 for x in retrieval['retained']:assert sha(HERE/'remote'/x['path'])==x['sha256']
 r=load(HERE/'remote/qualification/result.json');p=load(HERE/'remote/qualification/platform.json');life=load(HERE/'remote/package-lifecycle.json');worker=load(HERE/'remote/package-worker.json')
 assert life['worker']==worker and life['asset_sha256']==worker['asset_sha256']==build['assets']
 assert life['identity']==worker['identity']==build['identity']
 assert life['qualified'] is False and worker['qualified'] is False
 assert life['worker_returncode']==worker['process']['returncode']==2
 assert worker['process']['reaped'] is True and life['package_cleanup_complete'] is True
 assert r['cleanup_complete'] is True and r['cases']==[] and r['model_loaded'] is False
 assert r['error']=='RuntimeError:TARGET_ACCELERATOR_NOT_OBSERVED:ACCELERATOR_NAME_MISMATCH'
 acc=p['accelerator'];assert acc['returncode']==0 and acc['status']=='completed' and acc['reaped'] is True
 rows=list(csv.reader(acc['stdout'].splitlines()));assert len(rows)==1 and len(rows[0])==4
 gpu,uuid,driver,memory=[x.strip() for x in rows[0]]
 assert gpu=='Tesla P100-PCIE-16GB' and acc['validation_error']=='ACCELERATOR_NAME_MISMATCH'
 assert all(x['returncode']==0 and x['stdout']=='LOAD_OK\n' for x in acc['dependency_checks'].values())
 quotas=[]
 for name in ('quota-before.json','quota-after.json'):
  q=load(HERE/name);assert q['returncode']==0
  quotas.append(next(x for x in json.loads(q['stdout']) if x['resource']=='GPU'))
 delta=Decimal(quotas[1]['used'][:-1])-Decimal(quotas[0]['used'][:-1]);assert 0<=delta<=Decimal('.5')
 verification=load(HERE/'verification/VERIFICATION.json');assert verification['verified'] is True
 assert verification['package']['build']==build
 out={**report,'qualification_outcome':'FAIL_CLOSED_WRONG_ACTUAL_GPU; TARGET_CONTAINMENT_NOT_EXERCISED','strict_prelaunch_provenance_rejection':strict_rejection,'requested_kernel':build['kernel'],'returned_kernel':kernel,'url':'https://www.kaggle.com/code/'+kernel,'identity_deviation':'Provider generated title-derived slug; explicitly reconciled to sole launch, version and complete notebook JSON; not a pass under original literal target','notebook_raw_bytes_equal':sha(PACKAGE/'qualification.ipynb')==sha(HERE/'server-notebook.ipynb'),'notebook_complete_json_and_executable_bytes_equal':True,'wire_canonical_sha256':wire['wire_notebook_sha256'],'first_failing_boundary':r['error'],'actual_gpu':{'name':gpu,'uuid':uuid,'driver':driver,'memory':memory},'requested_and_server_machine_shape':server['machine_shape'],'requested_and_server_image':server['docker_image'],'loader_query_and_NVML_CUDA_dlopen':'PASS_ACTUAL_REMOTE','remote_seals_rootfs_canaries_parentdeath':'NOT_EXERCISED; no case runner or rootfs created','trusted_assessor_reaped':True,'package_cleanup_complete':True,'model_loaded':False,'local_mapped_tests':verification['checks']['mapped-root-tests'],'local_additional_approval_tests':4,'local_actual_notebook_normal_and_cached_cases_each':16,'gpu_quota_before':quotas[0],'gpu_quota_after':quotas[1],'gpu_quota_display_delta_hours':str(delta),'first_durable_terminal_observation_seconds':status['since_launch_s'],'assessor_elapsed_seconds':r['elapsed_s'],'provider_timeout_seconds_requested':300,'provider_timeout_server_attested':False,'billing_limitation':'Rounded account quota, not exact per-job GPU billing or runtime. No exact cancellation claim.','push_count':1,'retry_count':0,'owned_remote_job_terminal':status['status'],'permitted_next_action':'Stop. No inference, fallback, repair or second launch authorized by this result.'}
 (HERE/'AUDIT.json').write_text(json.dumps(out,indent=2)+'\n')
 (HERE/'RESULT.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps(out,indent=2))
 return 2
if __name__=='__main__':raise SystemExit(main())
