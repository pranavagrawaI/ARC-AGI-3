"""Offline audit for either strict wrong-GPU abort or full 16-case result."""
import csv, hashlib, json, re
from decimal import Decimal
from pathlib import Path
import audit_remote
from verify_package import HERE, PACKAGE, verify


def load(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    verify()
    build=load(PACKAGE/'BUILD.json')
    launch=load(HERE/'launch.json')
    assert launch['returncode']==0
    match=re.search(r'Kernel version (\d+) successfully pushed\.  Please check progress at https://www.kaggle.com/code/(\S+)',launch['stdout'])
    assert match
    version=int(match[1]); kernel=match[2]
    assert version==1
    assert launch['arguments']==['kernels','push','--path',str(PACKAGE),'--timeout','300','--accelerator','NvidiaRtxPro6000']
    status=load(HERE/'status-exact-version-final.json')
    server=load(HERE/'SERVER.json')
    server_receipt=load(HERE/'SERVER-RECEIPT.json')
    assert server_receipt['returncode']==0 and server_receipt['returned_ref']==kernel and server_receipt['returned_version']==version
    assert server_receipt['source_sha256']==sha(HERE/'server-notebook.ipynb')
    assert server['competition_sources']==['arc-prize-2026-arc-agi-3']
    assert all(server[k]==[] for k in ('dataset_sources','kernel_sources','model_sources'))
    local_nb=load(PACKAGE/'qualification.ipynb'); server_nb=load(HERE/'server-notebook.ipynb')
    assert local_nb==server_nb
    canonical=json.dumps(server_nb).encode()
    wire={'local_notebook_sha256':sha(PACKAGE/'qualification.ipynb'),'server_raw_sha256':sha(HERE/'server-notebook.ipynb'),
          'wire_notebook_sha256':hashlib.sha256(canonical).hexdigest(),'complete_parsed_notebook_equal':True}
    (HERE/'WIRE-CONTRACT.json').write_text(json.dumps(wire,indent=2)+'\n')
    contract={**build,'kernel':kernel,'version':version,'image':server['docker_image'],'notebook_sha256':wire['wire_notebook_sha256']}
    report=audit_remote.audit_remote(HERE/'remote/qualification',contract,status,server,canonical)
    retrieval=load(HERE/'RETRIEVAL.json')
    assert retrieval['kernel']==kernel and retrieval['version']==version
    result=load(HERE/'remote/qualification/result.json')
    platform=load(HERE/'remote/qualification/platform.json')
    lifecycle=load(HERE/'remote/package-lifecycle.json')
    worker=load(HERE/'remote/package-worker.json')
    expected={'qualification/result.json','qualification/platform.json','package-lifecycle.json','package-worker.json'}
    expected.update('qualification/'+row['mode']+'-'+row['name']+'-rootfs.json' for row in result['cases'])
    assert {x['path'] for x in retrieval['retained']}==expected
    for item in retrieval['retained']: assert sha(HERE/'remote'/item['path'])==item['sha256']
    assert lifecycle['worker']==worker and lifecycle['asset_sha256']==worker['asset_sha256']==build['assets']
    assert lifecycle['identity']==worker['identity']==build['identity']
    assert lifecycle['package_cleanup_complete'] is True and result['cleanup_complete'] is True
    assert result['model_loaded'] is False
    accelerator=platform['accelerator']
    assert accelerator['returncode']==0 and accelerator['status']=='completed' and accelerator['reaped'] is True
    rows=list(csv.reader(accelerator['stdout'].splitlines(),skipinitialspace=True))
    assert len(rows)==1 and len(rows[0])==4
    gpu,uuid,driver,memory=[x.strip() for x in rows[0]]
    requested_observed='observed_device' in accelerator and not accelerator.get('validation_error')
    if requested_observed:
        assert result.get('observed_accelerator')==accelerator['observed_device']
        assert report['executed']==report['passed']==report['required']==16 and report['failed']==report['unexercised']==0
        assert report['qualified'] is True and lifecycle['qualified'] is True and worker['qualified'] is True
        outcome='RIGHT_GPU_ALL_16_CONTAINMENT_CASES_PASSED'
    else:
        assert accelerator.get('validation_error')=='ACCELERATOR_NAME_MISMATCH'
        assert result['cases']==[] and report['executed']==0 and report['unexercised']==16
        assert report['qualified'] is False and lifecycle['qualified'] is False and worker['qualified'] is False
        assert result['error']=='RuntimeError:TARGET_ACCELERATOR_NOT_OBSERVED:ACCELERATOR_NAME_MISMATCH'
        outcome='FAIL_CLOSED_WRONG_ACTUAL_GPU; TARGET_CONTAINMENT_NOT_EXERCISED'
    quotas=[]
    for name in ('quota-before.json','quota-after.json'):
        receipt=load(HERE/name); assert receipt['returncode']==0
        quotas.append(next(x for x in json.loads(receipt['stdout']) if x['resource']=='GPU'))
    delta=Decimal(quotas[1]['used'][:-1])-Decimal(quotas[0]['used'][:-1])
    assert Decimal('0')<=delta<=Decimal('.5')
    out={**report,'qualification_outcome':outcome,'requested_kernel':build['kernel'],'returned_kernel':kernel,
         'url':'https://www.kaggle.com/code/'+kernel,'notebook_raw_bytes_equal':wire['local_notebook_sha256']==wire['server_raw_sha256'],
         'notebook_complete_json_and_executable_bytes_equal':True,'wire_canonical_sha256':wire['wire_notebook_sha256'],
         'first_failing_boundary':result.get('error'),'actual_gpu':{'name':gpu,'uuid':uuid,'driver':driver,'memory':memory},
         'requested_gpu_truly_observed':requested_observed,'requested_and_server_machine_shape':server['machine_shape'],
         'requested_competition_sources':['arc-prize-2026-arc-agi-3'],'server_competition_sources':server['competition_sources'],
         'competition_association_exact':True,'unintended_server_sources_absent':True,
         'requested_and_server_image':server['docker_image'],'loader_query_and_NVML_CUDA_dlopen':'PASS_ACTUAL_REMOTE',
         'remote_seals_rootfs_canaries_parentdeath':'PASSED_16_CASES' if report['qualified'] else 'NOT_EXERCISED; no case runner or rootfs created',
         'trusted_assessor_reaped':worker['process']['reaped'],'package_cleanup_complete':True,'model_loaded':False,
         'gpu_quota_before':quotas[0],'gpu_quota_after':quotas[1],'gpu_quota_display_delta_hours':str(delta),
         'first_durable_terminal_observation_seconds':status['since_launch_s'],'assessor_elapsed_seconds':result['elapsed_s'],
         'provider_timeout_seconds_requested':300,'provider_timeout_server_attested':False,
         'billing_limitation':'Rounded account quota, not exact per-job GPU billing or runtime. No exact cancellation claim.',
         'push_count':1,'retry_count':0,'owned_remote_job_terminal':status['status'],
         'permitted_next_action':'Stop. This authorization permits no further retry, fallback, repair, model loading or second launch.',
         'allocation_cause':'NOT_ESTABLISHED'}
    (HERE/'AUDIT.json').write_text(json.dumps(out,indent=2)+'\n')
    (HERE/'RESULT.json').write_text(json.dumps(out,indent=2)+'\n')
    md=f"""# Result\n\n- Classification: **{out['classification']}**\n- Qualification outcome: **{outcome}**\n- Exact run: [{kernel}/1](https://www.kaggle.com/code/{kernel})\n- Server competition association: exact `arc-prize-2026-arc-agi-3`; no dataset/kernel/model sources\n- Requested/server shape: `NvidiaRtxPro6000`; actual GPU: `{gpu}` (`{uuid}`)\n- Cases: required {out['required']}, executed {out['executed']}, passed {out['passed']}, failed {out['failed']}, unexercised {out['unexercised']}\n- Terminal status: `{out['terminal_status']}`; first durable terminal observation {out['first_durable_terminal_observation_seconds']:.3f}s after launch\n- Displayed GPU quota: {quotas[0]['used']} -> {quotas[1]['used']} (delta {delta}h; rounded display, not exact billing)\n- Pushes: 1; retries: 0; model loaded: false; package cleanup: verified\n\nThe runtime query, not requested/server metadata, determines actual hardware. The allocation cause is not established by these observations. No scheduler-fallback claim is made. No exact cancellation claim is made.\n"""
    (HERE/'RESULT.md').write_text(md)
    print(json.dumps(out,indent=2))
    return 0 if out['qualified'] else 2

if __name__=='__main__': raise SystemExit(main())
