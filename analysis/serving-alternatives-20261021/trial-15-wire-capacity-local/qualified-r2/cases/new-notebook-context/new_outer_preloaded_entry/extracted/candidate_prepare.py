"""Offline candidate setup/identity/exec, invoked only by the reviewed supervisor.

Model weights stay on the pinned Kaggle mount. We check their names/sizes and
small configuration hashes, not a second full 180GB weight read before startup.
"""
import csv
import io
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import candidate_library as library

HERE=Path(__file__).resolve().parent
PLAN=json.loads((HERE/'candidate-plan.json').read_text())
RUNTIME_ROOT=Path('/tmp/serving-wire-capacity15-runtime')


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def save(name,value):
    out=Path(os.environ['SCREEN_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
    (out/name).write_text(json.dumps(value,indent=2)+'\n')


def dataset(owner,slug):
    paths=[Path('/kaggle/input/datasets')/owner/slug,Path('/kaggle/input')/slug]
    found=[p for p in paths if p.is_dir()]
    if len(found)!=1:raise ValueError('missing/ambiguous exact mounted dataset '+owner+'/'+slug)
    return found[0]


def model_path():
    owner,slug,framework,instance,version=PLAN['model']['model_version'].split('/')
    paths=[Path('/kaggle/input/models')/owner/slug/framework.lower()/instance/version,
           Path('/kaggle/input')/slug/framework.lower()/instance/version]
    found=[p for p in paths if p.is_dir()]
    if len(found)!=1:raise ValueError('missing/ambiguous pinned model version mount')
    return found[0]


def bind():
    if not Path('/kaggle/input').is_dir():raise ValueError('actual setup is Kaggle-only')
    if sys.version_info[:2]!=(3,12):raise ValueError('candidate image requires host Python3.12 ABI')
    if os.environ.get('SCREEN_RUN_ID')!=PLAN['run_id']:raise ValueError('supervisor run identity missing')
    owner,slug=PLAN['runtime_dataset'].split('/')
    library.RUNTIME_DATASET=dataset(owner,slug)
    library.MODEL_PATH=model_path()
    library.IMAGE_RUNTIME_ROOT=RUNTIME_ROOT
    library.IMAGE_RUNTIME_CACHE_ROOT=RUNTIME_ROOT.with_name(RUNTIME_ROOT.name+'-cache')
    library.VLLM_DRAFT_VOCAB_FILE=HERE/'draft_vocab_32k.json'
    if digest(HERE/'candidate_library.py')!=PLAN['library_sha']:raise ValueError('candidate library changed')
    if digest(library.VLLM_DRAFT_VOCAB_FILE)!=PLAN['draft_vocab_sha']:raise ValueError('draft vocabulary changed')


def command():
    argv=library.build_vllm_server_command()
    argv[argv.index('--limit-mm-per-prompt')+1]=json.dumps(dict(image=58,video=0))
    template=HERE/'chat_template.normalized.jinja'
    if '--chat-template' in argv:argv[argv.index('--chat-template')+1]=str(template)
    else:argv+=['--chat-template',str(template)]
    return argv+['--enable-prompt-tokens-details']


def host_guard():
    inventory=subprocess.check_output(['nvidia-smi'],text=True,timeout=10)
    save('actual-nvidia-smi.json',dict(raw=inventory))
    query=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version,uuid',
                                   '--format=csv,noheader,nounits'],text=True,timeout=10)
    save('actual-gpu-query.json',dict(raw=query))
    rows=[row for row in csv.reader(io.StringIO(query)) if any(cell.strip() for cell in row)]
    if len(rows)!=1 or len(rows[0])!=4:raise ValueError('actual GPU query requires exactly one GPU/four fields')
    name,memory,driver,uuid=(value.strip() for value in rows[0])
    if name!='NVIDIA RTX PRO 6000 Blackwell Server Edition':raise ValueError('actual GPU identity mismatch')
    if not memory.isdigit() or int(memory)<90000:raise ValueError('actual GPU memory below90000MiB')
    if not driver or not uuid.startswith('GPU-'):raise ValueError('actual GPU driver/UUID identity missing')
    save('actual-gpu-query.json',dict(raw=query,name=name,memory_mib=int(memory),driver=driver,uuid=uuid))
    capability=inventory.split('CUDA Version:',1)[1].split()[0]
    if tuple(map(int,capability.split('.'))) < (13,0):raise ValueError('CUDA13 driver capability missing')
    meminfo=Path('/proc/meminfo').read_text();total=int(meminfo.split('MemTotal:',1)[1].split()[0])*1024
    raw=Path('/sys/fs/cgroup/memory.max').read_text().strip()
    limit=min(total,int(raw)) if raw.isdigit() else total
    free=shutil.disk_usage('/tmp').free
    save('host-headroom.json',dict(meminfo=meminfo,cgroup_memory_max=raw,effective_memory_bytes=limit,tmp_free_bytes=free))
    if limit<170*1024**3:raise ValueError('host memory below170GiB')
    if free<80*1024**3:raise ValueError('writable disk below80GiB')


def prepare():
    host_guard()
    if RUNTIME_ROOT.exists():raise ValueError('runtime root already exists; no retry/reuse')
    runtime=library.RUNTIME_DATASET
    if digest(runtime/'runtime-manifest.json')!=PLAN['runtime_manifest_sha']:raise ValueError('runtime manifest mismatch')
    manifest=library.read_runtime_manifest();library.validate_prefix_cache_runtime(manifest)
    if digest(runtime/manifest['patch']['applier'])!=PLAN['applier_sha']:raise ValueError('runtime applier source mismatch')
    if digest(runtime/'PATCH_IDENTITY.json')!=PLAN['patch_identity_sha']:raise ValueError('runtime patch identity source mismatch')
    if digest(runtime/manifest['patch']['artifact'])!=manifest['patch']['overlay_sha256']:raise ValueError('runtime overlay mismatch')
    layers={}
    for layer in manifest['selected_layers']:
        actual=digest(runtime/layer['file']);layers[layer['file']]=actual
        if actual!=layer['sha256']:raise ValueError('runtime layer source mismatch '+layer['file'])
    save('runtime-input-hashes.json',layers)
    files=PLAN['model']['files'];model=library.MODEL_PATH
    inventory={}
    for row in files:
        path=model/row['name'];size=path.stat().st_size
        if size!=int(row['size']):raise ValueError('model file size mismatch '+row['name'])
        inventory[row['name']]=size
    assets={}
    for name,expected in PLAN['model']['assets'].items():
        actual=digest(model/name);assets[name]=actual
        if actual!=expected['sha256']:raise ValueError('model processor/config identity mismatch '+name)
    for row in files:
        if row['name'].endswith(('.json','.jinja','.txt')) and int(row['size'])<=32*1024**2:
            assets[row['name']]=digest(model/row['name'])
    if digest(HERE/'chat_template.normalized.jinja')!=PLAN['normalized_template_sha']:raise ValueError('normalization template changed')
    save('model-source-identity.json',dict(version=PLAN['model']['model_version'],mount=str(model),
         files=inventory,asset_sha256=assets,full_weight_sha256_verified=False,
         source_binding='pinned Kaggle model version + all expected filenames/sizes + configuration hashes'))
    library.install_image_runtime()
    dist=RUNTIME_ROOT/manifest['dist_packages'];actual={}
    for row in manifest['patch']['overlay_files']:
        sha=digest(dist/row['target']);actual[row['target']]=sha
        if sha!=row['sha256']:raise ValueError('installed source mismatch '+row['target'])
    # Record exact frontend and scheduler bytes separately, without instrumenting them.
    names=set(PLAN['candidate_frontend_source_sha'])|{'vllm/v1/core/sched/scheduler.py','vllm/v1/request.py',
                 'vllm/entrypoints/openai/chat_completion/serving.py'}
    for name in sorted(names):
        actual[name]=digest(dist/name)
        destination=Path(os.environ['SCREEN_OUTPUT'])/'installed-source'/name
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(dist/name,destination)
        save('installed-source-identity.json',actual)
        expected=PLAN['candidate_frontend_source_sha'].get(name)
        if expected is not None and actual[name]!=expected:raise ValueError('installed frontend/source contract mismatch '+name)
    save('installed-source-identity.json',actual)
    env=library.build_image_runtime_env(manifest);argv=command()
    save('candidate-launch.json',dict(argv=argv,environment={k:v for k,v in env.items() if k.startswith(('VLLM_','CUDA_','TORCH_','PYTORCH_'))},
                                   native_overlay_enabled=False,image_max_pixels=409600,image_limit=58,sequences=10))
    save('prepared.json',dict(manifest_sha=PLAN['runtime_manifest_sha'],runtime_root=str(RUNTIME_ROOT),rate_comparison=False))


def identity():
    manifest=library.read_runtime_manifest();env=library.build_image_runtime_env(manifest)
    code="import json,torch,vllm,transformers,sys;print(json.dumps({'python':sys.version,'vllm':vllm.__version__,'torch':torch.__version__,'transformers':transformers.__version__,'cuda_build':torch.version.cuda}))"
    result=subprocess.run([sys.executable,'-c',code],env=env,check=True,capture_output=True,text=True,timeout=100)
    observed=json.loads(result.stdout.splitlines()[-1]);save('installed-package-identity.json',observed)
    if observed['vllm']!=manifest['vllm_version'] or observed['torch']!=manifest['torch_version']:
        raise ValueError('installed runtime version mismatch')


def main():
    bind();operation=sys.argv[1]
    if operation=='prepare':prepare()
    elif operation=='identity':identity()
    elif operation=='serve':
        if not (Path(os.environ['SCREEN_OUTPUT'])/'prepared.json').is_file():raise ValueError('missing preparation receipt')
        env=library.build_image_runtime_env(library.read_runtime_manifest())
        os.execvpe(sys.executable,command(),env)
    else:raise ValueError('unknown preparation operation')


if __name__=='__main__':main()
