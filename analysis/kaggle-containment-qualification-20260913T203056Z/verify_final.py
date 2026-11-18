"""Deterministic final evidence and package-integrity verification."""
import hashlib, json
from pathlib import Path
import verify_package
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]

def main():
    manifest=json.loads((HERE/'FINAL-MANIFEST.json').read_text())
    for relative,digest in manifest['files'].items():
        assert hashlib.sha256((REPO/relative).read_bytes()).hexdigest()==digest, relative
    result=json.loads((HERE/'RESULT.json').read_text())
    assert result==json.loads((HERE/'AUDIT.json').read_text())
    assert result['push_count']==1 and result['retry_count']==0
    assert result['requested_gpu_truly_observed'] is False
    assert (result['required'],result['executed'],result['passed'],result['failed'],result['unexercised'])==(16,0,0,0,16)
    assert result['actual_gpu']['name']=='Tesla P100-PCIE-16GB'
    assert result['terminal_status']=='ERROR' and result['provenance']=='EXACT_TARGET_RECONCILED'
    assert verify_package.verify()['verified'] is True
    retrieval=json.loads((HERE/'RETRIEVAL.json').read_text())
    assert {x['path'] for x in retrieval['retained']}=={'qualification/result.json','qualification/platform.json','package-lifecycle.json','package-worker.json'}
    processes=json.loads((HERE/'OWNED-PROCESSES.json').read_text())
    assert processes['active_owned_processes']==[]
    print(json.dumps({'evidence_integrity_verified':True,'files_verified':len(manifest['files']),
      'final_manifest_sha256':hashlib.sha256((HERE/'FINAL-MANIFEST.json').read_bytes()).hexdigest(),
      'qualified':result['qualified'],'actual_gpu':result['actual_gpu']['name'],'remote_executed':result['executed'],
      'remote_unexercised':result['unexercised'],'active_owned_processes':0},indent=2))

if __name__=='__main__': main()
