"""Build/check and extract the exact notebook payload. Never contacts Kaggle."""
import argparse
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile
import build_notebook as builder
from freeze_candidate import verify

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT/'analysis/deployment-probes/20260906-q6-summary-boundary'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    manifest=verify()
    notebook=builder.build(max_actions='100',worker_timeout_s=900,launcher_timeout_s=8400,
                           heading='Q6 summary boundary repair longer run — 100 environment actions, 7200s admission horizon, control stall arm')
    builder.validate(notebook,max_actions='100',launcher_timeout_s=8400)
    path=HERE/'probe.ipynb'
    if args.check:
        assert json.loads(path.read_text()) == notebook, 'Notebook differs from pinned source'
    else:
        path.write_text(json.dumps(notebook,indent=2)+'\n')
    OUT.mkdir(parents=True,exist_ok=True)
    extracted=OUT/'extracted'
    extracted.mkdir(exist_ok=True)
    code='\n'.join(''.join(c.get('source',[])) for c in notebook['cells'] if c['cell_type']=='code')
    payloads=[]
    for node in ast.walk(ast.parse(code)):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='b64decode' and isinstance(node.args[0],ast.Constant):
            raw=base64.b64decode(node.args[0].value)
            payloads.append(raw)
            if raw.startswith(b'PK'):
                with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                    for name in archive.namelist():
                        assert not Path(name).is_absolute() and '..' not in Path(name).parts
                        dest=extracted/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(archive.read(name))
    for name in ['exact_request_outer_runtime.py','adapter.py','shared_local_solver.py','run_official_agent.py','nooa_public_probe.py']:
        raw=(HERE/name).read_bytes()
        assert raw in payloads, f'{name} missing or stale outside zip'
        (extracted/name).write_bytes(raw)
    for name,digest in manifest['files'].items():
        assert hashlib.sha256((extracted/name).read_bytes()).hexdigest()==digest
    metadata=json.loads((HERE/'kernel-metadata.json').read_text())
    assert metadata['id']=='pranavagrawai/q6-summary-boundary-20260906'
    contract={'kernel':metadata['id'],'session_timeout_seconds':10800,'gameplay_admission_seconds':7200,'launcher_timeout_seconds':8400,
              'worker_timeout_seconds':900,'max_environment_actions':100,'contract_sha256':manifest['contract_sha256'],
              'maximum_pushes':1,'automatic_retries':0,'argv':['kaggle','kernels','push','--path',str(HERE),'--timeout','10800','--accelerator','NvidiaRtxPro6000'],
              'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [path,HERE/'kernel-metadata.json',HERE/'EXPERIMENT-CARD.md',HERE/'prepare.py',HERE/'verify_probe.py',HERE/'assess.py']}}
    launch=HERE/'LAUNCH.json'
    if args.check:
        assert json.loads(launch.read_text())==contract, 'Launch bytes drifted'
    else:
        launch.write_text(json.dumps(contract,indent=2)+'\n')
    print(json.dumps(contract,indent=2))


if __name__=='__main__':main()
