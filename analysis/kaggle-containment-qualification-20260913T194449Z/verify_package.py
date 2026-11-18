"""Verify exact notebook bootstrap, assets and deployment contract offline."""
import ast
import base64
import hashlib
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parents[1]/'deployment/kaggle'/HERE.name


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    spec=importlib.util.spec_from_file_location('candidate_build',PACKAGE/'build.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    build=json.loads((PACKAGE/'BUILD.json').read_text())
    assert build['identity']==HERE.name and build['launch_authorized'] is True
    assert set(build['assets'])==set(module.ASSET_NAMES)
    assert set(build['files'])=={'build.py','qualification.ipynb','kernel-metadata.json'}
    for name,h in build['files'].items(): assert sha(PACKAGE/name)==h, name
    for name,h in build['assets'].items(): assert sha(HERE/name)==h, name
    notebook=json.loads((PACKAGE/'qualification.ipynb').read_text())
    assert len(notebook['cells'])==1
    code=notebook['cells'][0]['source']; tree=ast.parse(code)
    assets=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='assets' for t in n.targets))
    assert set(assets)==set(module.ASSET_NAMES)
    for name,item in assets.items():
        raw=base64.b64decode(item['base64'],validate=True)
        assert raw==(HERE/name).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==item['sha256']==build['assets'][name]
        ast.parse(raw)
    expected=module.render(assets)
    assert code==expected, 'NOTEBOOK_BOOTSTRAP_DRIFT'
    metadata=json.loads((PACKAGE/'kernel-metadata.json').read_text())
    assert metadata['id']==module.ID and metadata['is_private'] is True
    assert metadata['enable_gpu'] is True and metadata['enable_tpu'] is False and metadata['enable_internet'] is False
    assert metadata['machine_shape']=='NvidiaRtxPro6000' and metadata['docker_image']==module.IMAGE
    assert metadata['docker_image_pinning_type']=='original'
    assert all(metadata[k]==[] for k in ('dataset_sources','kernel_sources','competition_sources','model_sources'))
    assert metadata['code_file']=='qualification.ipynb'
    return {'verified':True,'identity':HERE.name,'notebook_bootstrap_exact':True,'embedded_assets_exact':True,
            'metadata_contract_exact':True,'launch_authorized':True,'build':build}

if __name__=='__main__': print(json.dumps(verify(),indent=2))
