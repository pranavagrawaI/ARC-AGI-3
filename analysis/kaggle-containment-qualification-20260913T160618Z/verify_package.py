"""Reconcile build sources and exact upload metadata without provider mutation."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parents[1]/'deployment/kaggle'/HERE.name

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def verify():
    build=json.loads((PACKAGE/'BUILD.json').read_text())
    for name,h in build['files'].items():
        assert sha(PACKAGE/name)==h, ('build drift',name)
    for name,h in build['assets'].items():
        assert sha(HERE/name)==h, ('source drift',name)
    notebook=json.loads((PACKAGE/'qualification.ipynb').read_text())
    assert len(notebook['cells'])==1
    code=notebook['cells'][0]['source']
    tree=ast.parse(code)
    assets=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='assets' for t in n.targets))
    assert set(assets)=={'runner.py','probe.py'}
    for name,item in assets.items():
        data=base64.b64decode(item['base64'],validate=True)
        assert data==(HERE/name).read_bytes()
        assert hashlib.sha256(data).hexdigest()==item['sha256']==build['assets'][name]
        ast.parse(data)
    metadata=json.loads((PACKAGE/'kernel-metadata.json').read_text())
    assert metadata['id']=='pranavagrawai/containment-q-20260913t160618z'
    assert metadata['is_private'] is True and metadata['enable_gpu'] is True
    assert metadata['enable_tpu'] is False and metadata['enable_internet'] is False
    assert metadata['machine_shape']=='NvidiaRtxPro6000'
    assert metadata['docker_image']=='gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461'
    assert metadata['docker_image_pinning_type']=='original'
    assert all(metadata[k]==[] for k in ('dataset_sources','model_sources','competition_sources','kernel_sources'))
    assert metadata['code_file']=='qualification.ipynb'
    paths=[HERE/n for n in ('runner.py','probe.py','test_qualification.py','EXPERIMENT-CARD.md','verify_package.py','provider_contract.py','launch_once.py','monitor.py')]
    paths += [PACKAGE/n for n in ('build.py','BUILD.json','qualification.ipynb','kernel-metadata.json')]
    return {'identity':HERE.name,'kernel':metadata['id'], 'files':{str(p):sha(p) for p in paths},
            'checks':{'embedded_assets_exact':True,'metadata_exact':True,'build_hashes_reconciled':True},
            'provider_timeout_seconds':300,'gpu_hour_execution_bound':300/3600,'gpu_cap_margin_factor':.5/(300/3600)}

if __name__=='__main__':
    report=verify()
    if '--freeze' in sys.argv:
        with (HERE/'PRELAUNCH.json').open('x') as f: json.dump(report,f,indent=2); f.write('\n')
    elif (HERE/'PRELAUNCH.json').exists():
        assert json.loads((HERE/'PRELAUNCH.json').read_text())==report,'frozen manifest drift'
    print(json.dumps(report,indent=2))
