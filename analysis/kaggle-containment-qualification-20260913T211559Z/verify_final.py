"""Rerunnable final evidence, association, cases and package check."""
import hashlib,json
from pathlib import Path
import verify_package
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]

def load(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    manifest=load(HERE/'FINAL-MANIFEST.json')
    for rel,digest in manifest['files'].items(): assert sha(REPO/rel)==digest, rel
    result=load(HERE/'RESULT.json'); assert result==load(HERE/'AUDIT.json')
    assert result['qualified'] is True and result['qualification_outcome']=='RIGHT_GPU_ALL_16_CONTAINMENT_CASES_PASSED'
    assert (result['required'],result['executed'],result['passed'],result['failed'],result['unexercised'])==(16,16,16,0,0)
    assert result['actual_gpu']['name']=='NVIDIA RTX PRO 6000 Blackwell Server Edition'
    assert result['requested_gpu_truly_observed'] is True
    assert result['requested_competition_sources']==result['server_competition_sources']==['arc-prize-2026-arc-agi-3']
    assert result['competition_association_exact'] is True and result['unintended_server_sources_absent'] is True
    assert result['terminal_status']=='COMPLETE' and result['provenance']=='EXACT_TARGET_RECONCILED'
    assert result['package_cleanup_complete'] is True and result['remote_seals_rootfs_canaries_parentdeath']=='PASSED_16_CASES'
    assert result['push_count']==1 and result['retry_count']==0 and result['model_loaded'] is False
    assert verify_package.verify()['verified'] is True
    retrieval=load(HERE/'RETRIEVAL.json'); assert len(retrieval['retained'])==20
    for item in retrieval['retained']: assert sha(HERE/'remote'/item['path'])==item['sha256']
    assert load(HERE/'FINAL-STATUS-READBACK.json')['official_cli_status']=='KernelWorkerStatus.COMPLETE'
    assert load(HERE/'OWNED-PROCESSES.json')['active_owned_processes']==[]
    print(json.dumps({'evidence_integrity_verified':True,'files_verified':len(manifest['files']),
      'final_manifest_sha256':sha(HERE/'FINAL-MANIFEST.json'),'qualified':True,
      'competition_sources':['arc-prize-2026-arc-agi-3'],'actual_gpu':result['actual_gpu']['name'],
      'remote_cases':{'required':16,'executed':16,'passed':16,'failed':0,'unexercised':0},
      'active_owned_processes':0},indent=2))
if __name__=='__main__': main()
