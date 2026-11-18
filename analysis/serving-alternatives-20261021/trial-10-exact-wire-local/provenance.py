"""Bind the API contrast to actual supervisor, package and release receipts."""
import hashlib
import json
from pathlib import Path
import control_prepare

HERE=Path(__file__).resolve().parent

def read(path):return json.loads(Path(path).read_text())
def require(value,message):
    if not value:raise ValueError('package provenance: '+message)

def check(root,control,candidate):
    root=Path(root);config=read(root.parent/'config.json');contract=read(HERE/'capacity-contract.json')
    sha=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    require(sha==contract['config_sha256'],'runtime configuration drift')
    trace=list(map(json.loads,(root/'supervisor.jsonl').read_text().splitlines()))
    require(trace and [r['seq'] for r in trace]==list(range(len(trace))),'supervisor raw sequence')
    require(trace[0]['event']=='start' and trace[0]['config_sha256']==sha and trace[0]['run_id']==config['run_id'],'supervisor configuration identity')
    hardware=[r for r in trace if r['event']=='hardware_pass'];require(len(hardware)==1,'actual hardware receipt')
    for label,value in (('control',control),('candidate',candidate)):
        capture=value['capture'];package=root/(label+'-package')
        spawn=[r for r in trace if r['event']=='spawn' and r['phase']==label+'.server']
        require(len(spawn)==1 and str(spawn[0]['pgid'])==capture['server_pgid'],'server process group join')
        require(capture['boot_id']==trace[0]['boot_id'] and capture['session_id']==config['run_id'] and capture['gpu_uuid']==hardware[0]['uuid'],'same actual session/GPU')
        require(any(r['event']=='arm_pass' and r['arm']==label for r in trace),'arm phases did not qualify')
        require(any(r['event']=='cleanup' and r['pgid']==int(capture['server_pgid']) and r['remaining']==[] for r in trace),'owned server group not cleaned')
        release=read(root/(label+'-release.json'))
        require(release['pass'] is True and release['port8001_free'] is True and release['observed']['uuid']==capture['gpu_uuid'],'resource release receipt')
        if label=='control':
            plan=read(HERE/'control-plan.json');source=read(package/'control-source-identity.json');model=read(package/'control-model-identity.json')
            require(source['wheel_sha256']==plan['wheel']['sha256'] and source['lock_sha256']==plan['wheel']['requirements_lock_sha256'],'shipping wheel/lock')
            require(source['installed_source_sha256']==plan['installed_source_sha256'],'installed control source drift')
            require(model['target_version']==plan['target_version'] and model['draft_version']==plan['draft_version'],'control model versions')
            actual=model['inventory']['target']['assets_sha256']
            require(all(actual[k]==v for k,v in plan['assets_sha256'].items()),'control processor asset drift')
            prepared=read(package/'control-prepared.json')
            identity_health=read(package/'control-health-identity.json');serve_health=read(package/'control-health-serve.json')
            identity_values=control_prepare.verify_health_record(identity_health,'identity')
            serve_values=control_prepare.verify_health_record(serve_health,'serve')
            require(identity_values==serve_values,'control health identity/serve drift')
            require(Path(serve_values['runtime_executable']).resolve()==Path(prepared['python']).resolve(),'control health runtime interpreter')
            argv=prepared['argv'];target=Path(argv[argv.index('--model-path')+1]);draft=Path(argv[argv.index('--speculative-draft-model-path')+1])
            control_prepare.verify_argv(argv,target,draft,Path('/unused'))
        else:
            plan=read(HERE/'candidate-plan.json');source=read(package/'installed-source-identity.json');model=read(package/'model-source-identity.json')
            require(all(source[k]==v for k,v in plan['candidate_frontend_source_sha'].items()),'installed candidate source drift')
            require(model['version']==plan['model']['model_version'],'candidate model version')
            require(all(model['asset_sha256'][k]==v['sha256'] for k,v in plan['model']['assets'].items()),'candidate processor asset drift')
            launched=read(package/'candidate-launch.json')
            require(launched['image_max_pixels']==409600 and launched['image_limit']==58 and launched['sequences']==10 and launched['native_overlay_enabled'] is False,'candidate declared configuration drift')
    control_group=int(control['capture']['server_pgid'])
    cleanup=next(r for r in trace if r['event']=='cleanup' and r['pgid']==control_group and r['remaining']==[])
    candidate_prepare=next(r for r in trace if r['event']=='spawn' and r['phase']=='candidate.prepare')
    require(cleanup['seq']<candidate_prepare['seq'],'two-load overlap')
    return dict(bound=True,config_sha256=sha,source_model_processor_receipts=True,owned_groups_reaped=True,resource_release=True,
                limitations=['Version mounts and weight index/name/size checks; full checkpoint weight rehash not performed.','OS filesystem caches may carry temporal/order effects; no engine-only causal identification.'])
