"""Read-only local reducer over version1 archive and exact frozen client.

Default prints receipt; --receipt PATH writes only that derived receipt. No API,
network, model execution or Python tool-code execution is performed.
"""
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import zipfile
P=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(P));import transport
R=P/'remote-readback';ROOT=R/'archive-extracted';RUN=ROOT/'serving-candidate-feas7';OUT=RUN/'output'


def require(condition,message):
    if not condition:raise ValueError(message)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())


def reduce():
    archive=read(R/'archive-collection.json');archive_path=R/'output-v1-kernel140347143.zip'
    require(archive['version']==1 and archive['kernel_id']==140347143,'archive version identity')
    require(sha(archive_path)==archive['archive_sha256'],'archive digest')
    inventory={}
    with zipfile.ZipFile(archive_path) as zipped:
        require(zipped.testzip() is None,'archive CRC corruption')
        for item in zipped.infolist():
            if item.is_dir():continue
            file=ROOT/item.filename
            data=zipped.read(item)
            require(file.read_bytes()==data,'archive extraction differs '+item.filename)
            inventory[item.filename]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    require(len(inventory)==archive['extracted_files']==991,'archive completeness count')
    require(not archive['prior_overlap_differences'],'prior collection differs')
    freeze=read(P/'freeze.json')
    for name,expected in freeze['files'].items():require(sha(P/name)==expected,'frozen binding changed '+name)
    metadata=read(R/'metadata-latest.json');accepted=read(P/'launch-response.json');marker=read(P/'launch-attempt.json')
    require(metadata['id']==accepted['kernelId']==140347143 and metadata['currentVersionNumber']==accepted['versionNumber']==1,'accepted/remote identity')
    remote=(R/'source-latest.ipynb').read_bytes();armed=(P/'kernel/serving-candidate-feas7-20261023.ipynb').read_bytes()
    require(transport.cell_source(json.loads(remote))==transport.cell_source(json.loads(armed)),'remote wrapper cell changed')
    inner=transport.decode(remote);require(hashlib.sha256(inner).hexdigest()==marker['inner_notebook_sha'],'decoded inner changed')
    values=transport.literals(transport.cell_source(json.loads(inner)));report=read(P/'review-packet.json')
    require(read(RUN/'config.json')==values['CONFIG'],'remote config mismatch')
    require(read(RUN/'assignment.json')==values['ASSIGNMENT'],'remote assignment mismatch')
    require(values['ASSIGNMENT']['serialized_assignment']==marker['slot']['serialized_assignment'],'assignment not marker-bound')
    require(hashlib.sha256(json.dumps(values['CONFIG'],sort_keys=True).encode()).hexdigest()==report['config_sha256'],'config digest mismatch')
    require({n:r['sha256'] for n,r in values['PAYLOAD'].items()}==report['files'],'inner source set differs')
    for name,record in values['PAYLOAD'].items():
        require((RUN/name).read_bytes()==base64.b64decode(record['base64'],validate=True),'remote extracted source changed '+name)
        require(sha(RUN/name)==record['sha256'],'remote extracted source digest '+name)
    # Use exact frozen reducer code; independent raw length, corpus and count checks follow.
    spec=importlib.util.spec_from_file_location('frozen_case_reducer',P/'sources/client.py');client=importlib.util.module_from_spec(spec);spec.loader.exec_module(client)
    cases=read(P/'sources/cases.json')['cases'];rows=[];casefiles=[]
    for case in cases:
        folder=OUT/'candidate-functional'/case['key'].replace(':','-')
        expected={'request.json','wire.json','attempt.json','http.json','response.bin'}
        require({f.name for f in folder.iterdir() if f.is_file()}==expected,'missing/extra raw case evidence '+case['key'])
        request=(folder/'request.json').read_bytes();body=(folder/'response.bin').read_bytes();http=read(folder/'http.json')
        require(request==(P/'sources'/case['request_path']).read_bytes(),'actual request bytes differ from frozen full-image corpus')
        require(len(body)==http['response_bytes'],'raw response byte count mismatch')
        lengths=[int(v) for k,v in http['response_headers'] if k.lower()=='content-length']
        require(lengths==[len(body)],'HTTP Content-Length mismatch')
        row=client.assess(folder,case)
        require(row['ok'] and row['code_ok'] and row['prompt_count_matches_local'] is True,'raw case failed '+case['key'])
        raw=json.loads(body)
        code=[json.loads(call['function']['arguments'])['code'] for call in raw['choices'][0]['message'].get('tool_calls',[]) if call['function']['name']=='python']
        rows.append(dict(key=case['key'],images=case['images'],http_status=http['status'],complete=http['complete'],
                         prompt_tokens=row['usage']['prompt_tokens'],expected_prompt_tokens=case['local_prompt_tokens'],
                         completion_tokens=row['usage']['completion_tokens'],response_id=row['response_id'],code=code,
                         request_sha256=sha(folder/'request.json'),response_sha256=sha(folder/'response.bin'),response_bytes=len(body)))
        casefiles.extend(str(f.relative_to(ROOT)) for f in folder.iterdir() if f.is_file())
    require(len(rows)==6 and len(casefiles)==30 and len({row['response_id'] for row in rows})==6,'six cases/unique IDs/completeness')
    progress=read(OUT/'candidate-functional/progress.json');verdict=read(OUT/'candidate-functional/verdict.json');rawverdict=read(OUT/'candidate-functional/raw-readback-verdict.json')
    require(verdict['status']==rawverdict['status']=='FIT_FUNCTIONAL_ONLY' and verdict['completed']==6 and verdict['unattempted']==[],'remote summary disagrees')
    require(verdict['rate'] is None and verdict['adoption'] is False and verdict['rate_comparison_qualified'] is False,'unsupported rate/adoption claim')
    gpu=read(OUT/'actual-gpu-query.json');head=read(OUT/'host-headroom.json')
    require(gpu['name']=='NVIDIA RTX PRO 6000 Blackwell Server Edition' and gpu['memory_mib']>=90000,'actual hardware guard')
    require(head['effective_memory_bytes']>=170*1024**3 and head['tmp_free_bytes']>=80*1024**3,'actual host guard')
    plan=read(P/'sources/plan.json');manifest_path=P.parent/'trial-5-serving-screen/candidate-feasibility/runtime-manifest.reference.json'
    require(sha(manifest_path)==plan['runtime_manifest_sha'],'runtime reference identity')
    manifest=read(manifest_path)
    require(read(OUT/'runtime-input-hashes.json')=={row['file']:row['sha256'] for row in manifest['selected_layers']},'runtime layer identities')
    installed=read(OUT/'installed-source-identity.json')
    for row in manifest['patch']['overlay_files']:require(installed[row['target']]==row['sha256'],'installed overlay mismatch '+row['target'])
    for name,digest in plan['candidate_frontend_source_sha'].items():
        require(installed[name]==digest and sha(OUT/'installed-source'/name)==digest,'installed frontend mismatch '+name)
    model=read(OUT/'model-source-identity.json')
    require(model['version']==plan['model']['model_version'],'model version changed')
    require(model['files']=={row['name']:int(row['size']) for row in plan['model']['files']},'model names/sizes changed')
    for name,record in plan['model']['assets'].items():require(model['asset_sha256'][name]==record['sha256'],'model asset changed '+name)
    package=read(OUT/'installed-package-identity.json')
    require(package['vllm']==manifest['vllm_version'] and package['torch']==manifest['torch_version'],'runtime version changed')
    launch=read(OUT/'candidate-launch.json');require(launch['native_overlay_enabled'] is False and launch['image_max_pixels']==409600 and launch['image_limit']==58 and launch['sequences']==10,'engine launch settings changed')
    trace=[json.loads(line) for line in (OUT/'supervisor.jsonl').read_text().splitlines()]
    result=read(OUT/'result.json')
    require(result['pass_'] is True and result['surviving_groups']==[] and trace[-1]['event']=='terminal' and trace[-1]['pass_'] is True,'result/cleanup')
    require(all(event.get('remaining')==[] for event in trace if event['event']=='cleanup'),'remaining process group')
    require(sum(event['event']=='spawn' and event.get('phase')=='candidate.server' for event in trace)==1,'server load count')
    terminal=read(R/'priority-collection.json')
    require(terminal['status']['status']=='COMPLETE' and terminal['quota']['used_s']==60473.273 and terminal['quota']['reserved_s']==0,'terminal/quota identity')
    return dict(status='FIT_FUNCTIONAL_ONLY',kernel_ref=metadata['ref'],kernel_id=140347143,version=1,
                raw_reducer='exact frozen client.assess plus independent archive/source/body-length/identity checks',
                archive_sha256=archive['archive_sha256'],archive_bytes=archive['archive_bytes'],archive_files=len(inventory),raw_case_files=len(casefiles),
                inventory=inventory,cases=rows,sources_verified=15,frozen_bindings_verified=len(freeze['files']),
                runtime_layers_verified=len(manifest['selected_layers']),model_filenames_sizes_verified=len(model['files']),full_weight_sha256_verified=False,
                actual_gpu=gpu,host_headroom={k:head[k] for k in ('effective_memory_bytes','tmp_free_bytes')},installed_package=package,
                supervisor_elapsed_s=result['elapsed_seconds'],surviving_groups=[],terminal_status='COMPLETE',terminal_quota=terminal['quota'],
                raw_worker_log_artifact='remote-readback/output/serving-candidate-feas7-20261023.log',
                live_log_stream='Ended prematurely; error preserved; not used as verdict source',
                qualification='these six full-image long-prefix requests and short diagnostic replies only',
                rate=None,rate_comparison_qualified=False,quality_qualified=False,adoption=False,native_processor_parity_qualified=False)


if __name__=='__main__':
    result=reduce()
    if len(sys.argv)==3 and sys.argv[1]=='--receipt':Path(sys.argv[2]).write_text(json.dumps(result,indent=2)+'\n')
    else:print(json.dumps({k:v for k,v in result.items() if k!='inventory'},indent=2))
