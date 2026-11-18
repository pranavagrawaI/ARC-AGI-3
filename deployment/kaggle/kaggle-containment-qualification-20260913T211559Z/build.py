"""Offline-only build. This package has no launch command or authorization."""
import ast
import base64
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAME = 'kaggle-containment-qualification-20260913T211559Z'
SOURCE = HERE.parents[2]/'analysis'/NAME
ID = 'pranavagrawai/containment-q-20260913t211559z'
IMAGE = 'gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461'
ASSET_NAMES = ('runner.py','probe.py','accelerator.py','audit_remote.py')

# Runs only in a new -I -S interpreter. No notebook sys.modules or site hooks.
WORKER = '''import hashlib,json,os,sys
from pathlib import Path
root,output=map(Path,sys.argv[1:3])
identity=sys.argv[3]
sources=json.loads(sys.argv[4])
receipt={'identity':identity,'asset_sha256':sources,'qualified':False,'worker_pid':os.getpid(),'stage':'worker_imports'}
try:
    if set(p.name for p in root.iterdir()) != set(sources):
        raise RuntimeError('WORKER_ASSET_ALLOWLIST')
    for name,h in sources.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=h:
            raise RuntimeError('WORKER_ASSET_DRIFT')
    sys.path.insert(0,str(root))
    import probe, audit_remote, accelerator, runner
    runner.arm_parent_death(int(sys.argv[5]))
    for module in (probe,audit_remote,accelerator,runner):
        if Path(module.__file__).resolve()!=root/(module.__name__+'.py'):
            raise RuntimeError('WORKER_IMPORT_ORIGIN')
    receipt['stage']='assessor'
    entry="import runpy,sys; sys.path.insert(0,sys.argv[1]); import runner; runner.arm_parent_death(int(sys.argv[3])); p=sys.argv[1]+'/probe.py'; sys.argv=[p,sys.argv[2]]; runpy.run_path(p,run_name='__main__')"
    receipt['process']=probe.bounded([sys.executable,'-I','-S','-B','-c',entry,str(root),str(output),str(os.getpid())],
        timeout=165,output_cap=32768,env={'PATH':'/usr/local/bin:/usr/bin:/bin','LANG':'C'})
    receipt['stage']='receipt_audit'
    receipt['audit']=audit_remote.audit_directory(output,identity=identity,sources=sources)
    receipt['qualified']=(receipt['process']['status']=='completed' and receipt['process']['returncode']==0 and receipt['audit']['qualified'])
    receipt['stage']='complete' if receipt['qualified'] else 'qualification_failed'
except Exception as error:
    receipt['error']=type(error).__name__+':'+str(error)
finally:
    target=output.parent/'package-worker.json'
    temp=target.with_suffix('.tmp')
    temp.write_text(json.dumps(receipt,indent=2)+'\\n'); temp.replace(target)
raise SystemExit(0 if receipt['qualified'] else 2)
'''

BOOTSTRAP = '''import base64, hashlib, json, os, subprocess, sys, tempfile
from pathlib import Path
assets = ASSETS_VALUE
identity = IDENTITY_VALUE
worker = WORKER_VALUE
output = Path('/kaggle/working/qualification')
lifecycle_path = output.parent/'package-lifecycle.json'
lifecycle = {'identity': identity, 'stage': 'bootstrap', 'qualified': False,
             'asset_sha256': {n:v['sha256'] for n,v in assets.items()}}
with lifecycle_path.open('x') as stream:
    json.dump(lifecycle, stream, indent=2)
try:
    if os.environ.get('KAGGLE_IS_COMPETITION_RERUN', '').lower() in ('1','true'):
        raise RuntimeError('COMPETITION_RERUN_PROHIBITED')
    if output.exists() or (output.parent/'package-worker.json').exists():
        raise RuntimeError('STALE_QUALIFICATION_DIRECTORY')
    if set(assets) != {'runner.py','probe.py','accelerator.py','audit_remote.py'}:
        raise RuntimeError('ASSET_ALLOWLIST_MISMATCH')
    with tempfile.TemporaryDirectory(prefix='containment-package-') as temp:
        root = Path(temp)
        for name, item in assets.items():
            data = base64.b64decode(item['base64'], validate=True)
            if hashlib.sha256(data).hexdigest() != item['sha256']:
                raise RuntimeError('ASSET_HASH_MISMATCH')
            (root/name).write_bytes(data)
        lifecycle['stage']='dedicated_worker'
        # Worker emits no stdout; all assessor streams are bounded inside it.
        # No probe/audit imports occur in the notebook interpreter.
        process=subprocess.run([sys.executable,'-I','-S','-B','-c',worker,str(root),str(output),identity,json.dumps(lifecycle['asset_sha256']),str(os.getpid())],
            stdin=subprocess.DEVNULL,close_fds=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True,
            timeout=180,env={'PATH':'/usr/local/bin:/usr/bin:/bin','LANG':'C'})
        lifecycle['worker_returncode']=process.returncode
        lifecycle['worker_stderr']=process.stderr[:8192]
        receipt=json.loads((output.parent/'package-worker.json').read_text())
        if receipt['identity']!=identity or receipt['asset_sha256']!=lifecycle['asset_sha256']:
            raise RuntimeError('WORKER_RECEIPT_IDENTITY')
        lifecycle['worker']=receipt
        lifecycle['process']=receipt.get('process')
        lifecycle['audit']=receipt.get('audit')
        lifecycle['qualified']=(process.returncode==0 and receipt['qualified'] is True)
    lifecycle['package_cleanup_complete']=not root.exists()
    lifecycle['qualified']=lifecycle['qualified'] and lifecycle['package_cleanup_complete']
    lifecycle['stage']='complete' if lifecycle['qualified'] else 'qualification_failed'
except Exception as error:
    lifecycle['error']=type(error).__name__+':'+str(error)
finally:
    data=json.dumps(lifecycle,indent=2)+'\\n'
    temporary_receipt=lifecycle_path.with_suffix('.tmp')
    temporary_receipt.write_text(data); temporary_receipt.replace(lifecycle_path)
    print(data)
if not lifecycle['qualified']:
    raise RuntimeError('CONTAINMENT_NOT_QUALIFIED:'+lifecycle.get('error',lifecycle['stage']))
'''


def render(assets):
    return BOOTSTRAP.replace('ASSETS_VALUE',repr(assets)).replace('IDENTITY_VALUE',repr(NAME)).replace('WORKER_VALUE',repr(WORKER))


def build():
    assets = {}
    for name in ASSET_NAMES:
        raw = (SOURCE/name).read_bytes(); ast.parse(raw)
        assets[name] = {'sha256':hashlib.sha256(raw).hexdigest(), 'base64':base64.b64encode(raw).decode()}
    code = render(assets); ast.parse(code); ast.parse(WORKER)
    notebook = {'nbformat':4, 'nbformat_minor':5,
        'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},
        'cells':[{'cell_type':'code','id':'containment-qualification','metadata':{},'execution_count':None,'outputs':[],'source':code}]}
    metadata={'id':ID, 'title':'containment-q-20260913t211559z',
        'code_file':'qualification.ipynb','language':'python','kernel_type':'notebook',
        'is_private':True,'enable_gpu':True,'machine_shape':'NvidiaRtxPro6000',
        'docker_image':IMAGE,'docker_image_pinning_type':'original',
        'enable_internet':False,'enable_tpu':False,'dataset_sources':[],
        'kernel_sources':[],'competition_sources':['arc-prize-2026-arc-agi-3'],'model_sources':[]}
    (HERE/'qualification.ipynb').write_text(json.dumps(notebook,indent=1)+'\n')
    (HERE/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    manifest={'identity':NAME,'kernel':ID,'launch_authorized':True,
        'assets':{n:v['sha256'] for n,v in assets.items()},
        'files':{n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ('build.py','qualification.ipynb','kernel-metadata.json')}}
    (HERE/'BUILD.json').write_text(json.dumps(manifest,indent=2)+'\n'); print(json.dumps(manifest,indent=2))

if __name__=='__main__': build()
