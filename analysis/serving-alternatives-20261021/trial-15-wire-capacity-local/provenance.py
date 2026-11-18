"""Bind the API contrast to actual supervisor, package and release receipts."""
import hashlib
import json
from pathlib import Path
import control_prepare
import capacity_evidence as evidence
import artifacts,time

HERE=Path(__file__).resolve().parent

def read(path):return json.loads(Path(path).read_text())
def require(value,message):
    if not value:raise ValueError('package provenance: '+message)

def server_binding(root,trace,capture,label,config,gpu_uuid):
    package=Path(root)/(label+'-package')
    bindings=[r for r in trace if r['event']=='capacity_server_bound' and r['arm']==label]
    require(len(bindings)==1,'actual registered replay server binding')
    binding=read(root/(label+'-server-binding.json'));identity=binding['identity']
    require(all(bindings[0].get(k)==v for k,v in binding.items()),'server binding trace/file drift')
    require(str(identity['pgid'])==capture['server_pgid'] and str(identity['starttime'])==capture['server_start'] and identity['pid']==identity['pgid'],'actual server PGID/starttime join')
    require(capture['boot_id']==trace[0]['boot_id']==identity['boot_id'] and capture['session_id']==binding['session_id']==config['run_id'] and capture['gpu_uuid']==binding['gpu_uuid']==gpu_uuid,'same actual session/GPU')
    require(any(r['event']=='arm_pass' and r['arm']==label and capture['end']<=r['t'] for r in trace),'arm phases did not qualify')
    if label=='control':
        registration=read(package/'control-frontend-registration.json');native=binding['registered_native_binding']
        require(identity==registration['wrapper']==native['members'][str(identity['pid'])] and registration['frontend']==native['members'][str(native['frontend'])],'registered native wrapper/frontend join')
        spawns=[r for r in trace if r['event']=='native_spawn'];require(len(spawns)==1 and spawns[0]['pid']==identity['pid'] and spawns[0]['argv']==config['control']['server']['argv'],'actual native spawn')
    else:
        spawns=[r for r in trace if r['event']=='spawn' and r.get('phase')=='candidate.server']
        require(len(spawns)==1 and spawns[0]['pid']==identity['pid'] and spawns[0]['argv']==config['candidate']['server']['argv'],'actual candidate spawn')
    return binding

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
        binding=server_binding(root,trace,capture,label,config,hardware[0]['uuid'])
        if label=='control':
            plan=read(HERE/'control-plan.json');source=read(package/'control-source-identity.json');model=read(package/'control-model-identity.json')
            require(source['wheel_sha256']==plan['wheel']['sha256'] and source['lock_sha256']==plan['wheel']['requirements_lock_sha256'],'shipping wheel/lock')
            require(source['installed_source_sha256']==plan['installed_source_sha256'],'installed control source drift')
            require(model['target_version']==plan['target_version'] and model['draft_version']==plan['draft_version'],'control model versions')
            actual=model['inventory']['target']['assets_sha256']
            require(all(actual[k]==v for k,v in plan['assets_sha256'].items()),'control processor asset drift')
            prepared=read(package/'control-prepared.json')
            reg=read(package/'control-frontend-registration.json');declared=reg['contract'];native=binding['registered_native_binding']
            prepared_sha=hashlib.sha256((package/'control-prepared.json').read_bytes()).hexdigest()
            reg_sha=hashlib.sha256((package/'control-frontend-registration.json').read_bytes()).hexdigest()
            require(declared['run_id']==config['run_id'] and declared['prepared_sha256']==prepared_sha and declared['launch_argv']==prepared['argv'],'registered/prepared source/argv drift')
            registered_path=str(Path(declared['prepared_path']).with_name('control-frontend-registration.json'))
            require(native['source_files'][declared['prepared_path']]==prepared_sha and native['source_files'][registered_path]==reg_sha,'bound registered source artifacts drift')
            observed=hashlib.sha256(b''.join(v.encode()+b'\0' for v in declared['observed_argv'])).hexdigest()
            require(reg['frontend']['cmdline_sha256']==observed and reg['frontend']['exe']==declared['interpreter_realpath'],'actual registered frontend argv/interpreter')
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
    release=read(root/'release.json');fresh=read(root/'candidate-admission.json');final=read(root/'candidate-release.json')
    evidence.release_evidence(release,fresh,trace)
    require(control['capture']['end']<=release['start'],'release before control replay completed')
    dispatch=[r for r in trace if r['event']=='spawn' and r.get('phase')=='candidate.prepare']
    require(len(dispatch)==1 and fresh['sample']['end']<=dispatch[0]['t']<=fresh['sample']['end']+2,'candidate dispatch admission freshness')
    work_end=trace[0]['work_deadline'];evidence.final_resource_evidence(final,trace,work_end)
    require(candidate['capture']['end']<=final['start'],'candidate cleanup before replay completion')
    boundaries=[json.loads(s) for s in (root/'host-boundaries.jsonl').read_text().splitlines()]
    rows=[json.loads(s) for s in (root/'host-series.jsonl').read_text().splitlines()];series=read(root/'host-series.result.json')
    evidence.oom_evidence(boundaries,rows,series,trace,root)
    def verify(value):
        if isinstance(value,dict):
            if {'path','sha256','bytes','complete'}<=value.keys():artifacts.verify(root/'query-artifacts',value,time.monotonic()+60)
            else:
                for v in value.values():verify(v)
        elif isinstance(value,list):
            for v in value:verify(v)
    for value in (release,fresh,final,read(root/'query-compatibility.json')):verify(value)
    return dict(bound=True,config_sha256=sha,source_model_processor_receipts=True,owned_groups_reaped=True,resource_release=True,experimental_nonhost_admission=True,original140_report_only=True,
                limitations=['Version mounts and weight index/name/size checks; full checkpoint weight rehash not performed.','OS filesystem caches may carry temporal/order effects; no engine-only causal identification.'])
