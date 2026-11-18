"""Build only the two audited synthetic probe assets; never package the repo."""
import ast
import base64
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAME = 'kaggle-containment-qualification-20260913T150421Z'
SOURCE = HERE.parents[2]/'analysis'/NAME
ID = 'pranavagrawai/containment-q-20260913t150421z'


def build():
    assets = {}
    for name in ('runner.py', 'probe.py'):
        raw = (SOURCE/name).read_bytes()
        ast.parse(raw)
        assets[name] = {'sha256':hashlib.sha256(raw).hexdigest(), 'base64':base64.b64encode(raw).decode()}
    code = '''import base64, hashlib, json, os, subprocess, sys, tempfile
from pathlib import Path
if os.environ.get('KAGGLE_IS_COMPETITION_RERUN', '').lower() in ('1','true'):
    raise RuntimeError('COMPETITION_RERUN_PROHIBITED')
assets = ASSETS
with tempfile.TemporaryDirectory(prefix='containment-package-') as temp:
    root = Path(temp)
    for name, item in assets.items():
        data = base64.b64decode(item['base64'])
        if hashlib.sha256(data).hexdigest() != item['sha256']:
            raise RuntimeError('ASSET_HASH_MISMATCH')
        (root/name).write_bytes(data)
    process = subprocess.run([sys.executable, '-B', str(root/'probe.py'), '/kaggle/working/qualification'],
                             stdin=subprocess.DEVNULL, close_fds=True, capture_output=True,
                             text=True, timeout=180, env={'PATH':'/usr/local/bin:/usr/bin:/bin','LANG':'C'})
    print('PROBE_RETURN_CODE', process.returncode)
    print(process.stdout[:16384])
    print(process.stderr[:16384])
    # Notebook COMPLETE is not qualification PASS. Preserve unsupported receipts.
    result = Path('/kaggle/working/qualification/result.json')
    if not result.exists():
        raise RuntimeError('MISSING_QUALIFICATION_RECEIPT')
'''.replace('ASSETS', repr(assets))
    ast.parse(code)
    notebook = {'nbformat':4, 'nbformat_minor':5,
        'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},
        'cells':[{'cell_type':'code','id':'containment-qualification','metadata':{},'execution_count':None,'outputs':[],'source':code}]}
    metadata={'id':ID, 'title':'Containment q 20260913T150421Z',
              'code_file':'qualification.ipynb','language':'python','kernel_type':'notebook',
              'is_private':True,'enable_gpu':True,'machine_shape':'NvidiaRtxPro6000',
              'docker_image':'gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461',
              'docker_image_pinning_type':'original',
              'enable_internet':False,'enable_tpu':False,'dataset_sources':[],
              'kernel_sources':[],'competition_sources':[],'model_sources':[]}
    (HERE/'qualification.ipynb').write_text(json.dumps(notebook,indent=1)+'\n')
    (HERE/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    manifest={'identity':NAME,'kernel':ID,'assets':{n:v['sha256'] for n,v in assets.items()},
              'files':{n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ('build.py','qualification.ipynb','kernel-metadata.json')}}
    (HERE/'BUILD.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__': build()
