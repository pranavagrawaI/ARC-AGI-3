"""Pinned shipping setup, with a foreground process group instead of detachment."""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import registration
import candidate_prepare as shared

HERE=Path(__file__).resolve().parent
PLAN=json.loads((HERE/'control-plan.json').read_text())
PREFIX=Path('/tmp/sgl-intel')

def model(version):
    owner,slug,framework,instance,number=version.split('/')
    paths=[Path('/kaggle/input/models')/owner/slug/framework/instance/number,Path('/kaggle/input')/slug/framework/instance/number]
    found=[p for p in paths if p.is_dir()]
    if len(found)!=1:raise ValueError('missing/ambiguous pinned control model '+version)
    return found[0]

def bind():
    if sys.version_info[:2]!=(3,12) or os.environ.get('SCREEN_RUN_ID')!=PLAN['run_id']:raise ValueError('control ABI/run identity')
    return model(PLAN['target_version']),model(PLAN['draft_version']),shared.dataset('dfranzen','pennyroyal-v253')

def verify_argv(argv,target,draft_view,wheelhouse):
    reference=json.loads((HERE/'control-argv.reference.json').read_text())
    normalized=list(argv);reference=list(reference)
    # Mounted directory aliases and the path-hashed draft view may differ; only
    # these path operands are normalized, all numerical/semantic arguments exact.
    for flag,value in (('--model-path',str(target)),('--speculative-draft-model-path',str(draft_view))):
        if normalized[normalized.index(flag)+1]!=value:raise ValueError('control argument identity '+flag)
        reference[reference.index(flag)+1]=value
    for flag in ('--chat-template','--speculative-token-map'):
        actual=Path(normalized[normalized.index(flag)+1]);expected=Path(reference[reference.index(flag)+1])
        if actual.name!=expected.name:raise ValueError('control asset argument '+flag)
        reference[reference.index(flag)+1]=str(actual)
    if normalized!=reference:raise ValueError('shipping argv drift')

def prepare():
    target,draft,wheelhouse=bind();shared.host_guard()
    if PREFIX.exists():raise ValueError('control prefix already exists; refuse reuse/retry')
    wheel=PLAN['wheel']
    wheels=list(wheelhouse.rglob(wheel['filename']));locks=list(wheelhouse.rglob('requirements.lock'))
    if len(wheels)!=1 or len(locks)!=1:raise ValueError('ambiguous pinned bundle')
    if shared.digest(wheels[0])!=wheel['sha256'] or shared.digest(locks[0])!=wheel['requirements_lock_sha256']:raise ValueError('control wheel/lock identity')
    assets={name:shared.digest(target/name) for name in PLAN['assets_sha256']}
    if assets!=PLAN['assets_sha256']:raise ValueError('control model/processor assets changed')
    extraction=PLAN['notebook_extract'];setup=HERE/'control_notebook_setup.py'
    if shared.digest(setup)!=extraction['setup']['sha256']:raise ValueError('setup extraction changed')
    prefetch=HERE/'control_prefetch.py'
    if shared.digest(prefetch)!=extraction['prefetch']['sha256']:raise ValueError('prefetch extraction changed')
    pre={};exec(compile(prefetch.read_text(),str(prefetch),'exec'),pre)
    pre['precache'](str(wheelhouse))
    ns=dict(MODEL_DIR=str(target),DRAFT_MODEL_DIR=str(draft),WHEELHOUSE_DIR=str(wheelhouse),
            WORKING_DIR=os.environ['SCREEN_OUTPUT'],SERVED_MODEL_NAME='flashnext',SERVED_MODEL_PORT=8001)
    exec(compile(setup.read_text(),str(setup),'exec'),ns)
    verify_argv(ns['args'],target,ns['DRAFT_VIEW'],wheelhouse)
    site=next(PREFIX.glob('venv/lib/python*/site-packages'));installed={}
    for name,expected in PLAN['installed_source_sha256'].items():
        actual=shared.digest(site/name);installed[name]=actual
        if actual!=expected:raise ValueError('installed shipping source mismatch '+name)
    source_identity=dict(wheel_sha256=shared.digest(wheels[0]),lock_sha256=shared.digest(locks[0]),installed_source_sha256=installed)
    shared.save('control-source-identity.json',source_identity)
    # Record all small draft/target configs, actual shard inventory. Version
    # mounts + index coverage + asset hashes; no full 185GB weight rehash claim.
    inventory={}
    for label,directory in (('target',target),('draft',Path(ns['DRAFT_VIEW']))):
        index,shards=ns['indexed_shards'](str(directory))
        inventory[label]={'mount':str(directory),'index_sha256':shared.digest(directory/'model.safetensors.index.json'),
                          'shards':{name:(directory/name).stat().st_size for name in shards},
                          'assets_sha256':{p.name:shared.digest(p) for p in directory.iterdir() if p.is_file() and p.suffix in ('.json','.jinja') and p.stat().st_size<32*1024**2}}
    shared.save('control-model-identity.json',dict(target_version=PLAN['target_version'],draft_version=PLAN['draft_version'],inventory=inventory,full_weight_sha256_verified=False))
    # Persist only runtime overrides, never the inherited host environment.
    overrides={k:v for k,v in ns['env'].items() if os.environ.get(k)!=v}
    state=dict(argv=ns['args'],env_overrides=overrides,python=ns['PYTHON'],target=str(target),draft=str(draft),configuration=ns['CFG'])
    shared.save('control-prepared.json',state)

def state():
    value=json.loads((Path(os.environ['SCREEN_OUTPUT'])/'control-prepared.json').read_text())
    env=dict(os.environ);env.pop('PYTHONPATH',None);env.update(value['env_overrides']);value['env']=env
    return value

HEALTH_CODE = """import hashlib,json,os,sys
from pathlib import Path
from sglang.srt import environ
print(json.dumps({'raw':{k:os.environ.get(k) for k in ('SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION','SGLANG_HEALTH_CHECK_TIMEOUT')},
'effective':{'generation_enabled':environ.envs.SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION.get(),'timeout_s':int(os.getenv('SGLANG_HEALTH_CHECK_TIMEOUT','20'))},
'native_environ_sha256':hashlib.sha256(Path(environ.__file__).read_bytes()).hexdigest(),'runtime_executable':sys.executable}))
"""

def verify_health_record(record, phase):
    policy=PLAN['health_policy'];effective=record['effective']
    if (record.get('phase')!=phase or record['raw']!=policy['raw'] or effective!=policy['effective']
            or type(effective['generation_enabled']) is not bool or type(effective['timeout_s']) is not int
            or record['native_environ_sha256']!=policy['native_environ_sha256']
            or not isinstance(record.get('runtime_executable'),str) or not record['runtime_executable']):
        raise ValueError('control health parsed configuration/source mismatch')
    return {k:record[k] for k in ('raw','effective','native_environ_sha256','runtime_executable')}

def health_probe(data, phase):
    policy=PLAN['health_policy']
    if {k:data['env'].get(k) for k in policy['raw']}!=policy['raw']:
        raise ValueError('control health explicit environment mismatch')
    result=subprocess.run([data['python'],'-c',HEALTH_CODE],env=data['env'],capture_output=True,text=True,check=True,timeout=10)
    record=json.loads(result.stdout.splitlines()[-1]);record['phase']=phase
    verify_health_record(record,phase)
    if Path(record['runtime_executable']).resolve()!=Path(data['python']).resolve():
        raise ValueError('control health interpreter mismatch')
    return record

def identity():
    data=state();health=health_probe(data,'identity')
    code="import json,sglang,torch,transformers;print(json.dumps({'sglang':sglang.__version__,'torch':torch.__version__,'transformers':transformers.__version__}))"
    p=subprocess.run([data['python'],'-c',code],env=data['env'],capture_output=True,text=True,check=True,timeout=100)
    shared.save('control-package-identity.json',json.loads(p.stdout.splitlines()[-1]))
    shared.save('control-health-identity.json',health)

def serve():
    data=state();health=health_probe(data,'serve')
    identity_record=json.loads((Path(os.environ['SCREEN_OUTPUT'])/'control-health-identity.json').read_text())
    if verify_health_record(identity_record,'identity')!=verify_health_record(health,'serve'):
        raise ValueError('control health identity/serve mismatch')
    shared.save('control-health-serve.json',health)
    prefetch=HERE/'control_prefetch.py'
    if shared.digest(prefetch)!=PLAN['notebook_extract']['prefetch']['sha256']:raise ValueError('prefetch source changed')
    ns={};exec(compile(prefetch.read_text(),str(prefetch),'exec'),ns)
    # Both this thread and server child are owned by the supervisor PGID. The
    # original model prefetch delay/threads and server argv/env stay exact.
    thread=threading.Thread(target=ns['precache'],args=(data['target'],data['draft']),kwargs={'delay':60,'threads':3})
    thread.start()
    child=subprocess.Popen(data['argv'],env=data['env'],stdin=subprocess.DEVNULL,start_new_session=False)
    registration.emit_control(child.pid,data,PLAN,PREFIX,Path(os.environ['SCREEN_OUTPUT']),min(float(os.environ['SCREEN_OVERALL_DEADLINE_MONOTONIC']),time.monotonic()+10))
    result=child.wait();thread.join();raise SystemExit(result)

if __name__=='__main__':
    operation=sys.argv[1]
    if operation=='prepare':prepare()
    elif operation=='identity':identity()
    elif operation=='serve':serve()
    else:raise ValueError('unknown operation')
