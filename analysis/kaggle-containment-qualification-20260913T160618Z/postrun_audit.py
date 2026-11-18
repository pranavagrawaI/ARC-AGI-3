"""Separate post-run audit; no mutation of frozen candidate or provider."""
import ast
import base64
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import probe
import runner
import tempfile
from verify_package import verify, HERE, PACKAGE


def main():
    load=lambda p: json.loads(p.read_text())
    frozen=load(HERE/'PRELAUNCH.json')
    assert verify()==frozen
    server=load(HERE/'SERVER-VERSION.json')
    assert server['current_version_number']==1 and server['ref']==frozen['kernel']
    status=load(HERE/'status-exact-version-final.json')
    assert status['returncode']==0 and '/1 has status "KernelWorkerStatus.COMPLETE"' in status['stdout']
    local=load(PACKAGE/'qualification.ipynb')['cells'][0]['source']
    pulled=load(next((HERE/'server-metadata').glob('*.ipynb')))['cells'][0]['source']
    assert local==pulled
    tree=ast.parse(pulled)
    assets=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='assets' for t in n.targets))
    for name,item in assets.items():
        assert base64.b64decode(item['base64'])==(HERE/name).read_bytes()
    retrieval=load(HERE/'RETRIEVAL.json')
    assert {r['path'] for r in retrieval['retained']}=={'qualification/result.json','qualification/platform.json'}
    for row in retrieval['retained']:
        assert hashlib.sha256((HERE/'remote'/row['path']).read_bytes()).hexdigest()==row['sha256']
    result=load(HERE/'remote/qualification/result.json')
    platform=load(HERE/'remote/qualification/platform.json')
    acc=platform['accelerator']
    assert acc['path']=='/opt/bin/nvidia-smi' and acc['returncode']==12
    assert 'couldn\'t find libnvidia-ml.so' in acc['stdout'] and acc['stderr']==''
    assert result['error']=='RuntimeError:TARGET_ACCELERATOR_NOT_OBSERVED'
    assert result['cases']==[] and result['completed_cases']==result['passed_cases']==0
    assert result['cleanup_complete'] is True and result['model_loaded'] is False
    assert not list((HERE/'remote/qualification').glob('*-rootfs.json'))
    # Failure-specific regression uses only this observed nonsecret result.
    with tempfile.TemporaryDirectory() as temp:
        out=Path(temp)/'result'
        with patch.object(probe,'snapshot',return_value={'accelerator':acc}), patch.object(runner,'qualify'), patch.object(probe,'rootfs') as rootfs:
            assert probe.main(out)==2
            rootfs.assert_not_called()
        assert load(out/'result.json')['error']==result['error']
    audit=load(HERE/'AUDIT.json')
    output={
        'assessment':'PRECONDITION_ABORT; REMOTE_CONTAINMENT_NOT_EXERCISED',
        'first_failing_boundary':'Trusted nvidia-smi execution cannot load NVIDIA management library',
        'exact_error':result['error'], 'accelerator_observation':acc,
        'physical_gpu_identity':'NOT_ESTABLISHED',
        'requested_accelerator':'NvidiaRtxPro6000','server_reported_accelerator':server['machine_shape'],
        'requested_and_server_reported_image':server['docker_image'],
        'runtime_digest_attestation':'NOT_AVAILABLE; provider metadata only',
        'remote_cases':{'required':16,'executed':0,'passed':0,'failed':0,'not_exercised':16},
        'remote_sealing_parentdeath_and_cleanup_of_created_rootfs':'NOT_EXERCISED; no rootfs created',
        'source_and_server_notebook_exact':True,'retained_receipt_hashes_verified':True,
        'observed_driver_failure_regression':'PASS: guard rejects before rootfs',
        'mount_evidence':[l for l in platform['mountinfo'].splitlines() if 'nvidia' in l],
        'diagnosis':'Absolute binary discovery was fixed and execution is proven by exit12. The driver library is unavailable to this sanitized invocation. Mountinfo shows /usr/local/nvidia/lib64; exact library presence and a corrective loader setting were not tested. This is not evidence of GPU absence or chroot failure.',
        'gpu_quota_display_delta_hours':audit['quota_display_delta_gpu_hours'],
        'monitor_terminal_elapsed_seconds':audit['since_launch_terminal_seconds'],
        'assessor_elapsed_seconds':result['elapsed_s'],
        'retry_count':0,'permitted_action':'Stop; no inference or further launch authorized by this result',
        'metadata_readback_limitation':'Versioned pull /1 returned403; unversioned pull succeeded and current_version_number1 plus exact source match reconcile it with sole launch.'}
    (HERE/'POST-RUN-DIAGNOSIS.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))

if __name__=='__main__': main()
