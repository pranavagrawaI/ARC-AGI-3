"""Build/check the authorized two-arm offline competition-attached notebook."""
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

HERE=Path(__file__).resolve().parent
OUT=HERE.parents[2]/'analysis/deployment-probes/20260907-designer-player-reasoning-notebook'

def build():
    nb=builder.build(max_actions='30',worker_timeout_s=1200,launcher_timeout_s=3000,
        max_tokens='32768',llm_timeout_s=1200,request_timeout_s=180,
        extra_body='{"chat_template_kwargs":{"enable_thinking":true},"reasoning":{"enabled":true}}',
        heading='# Designer–player Kaggle comparison: 32k reasoning allowance\n\nLoop versus loop + designer-player policy. 32,768 total output tokens per request with natural stopping; measure actual use. Verified 262,144-token model context. Fixed FP8 baseline, MTP off. Public ls20 only; no submission or automatic promotion.')
    builder.validate(nb,max_actions='30',launcher_timeout_s=3000,
        extra_body='{"chat_template_kwargs":{"enable_thinking":true},"reasoning":{"enabled":true}}')
    body=''.join(nb['cells'][3]['source']).split('submission = pd.DataFrame(')[0]
    body=body.replace("os.environ['TYCHO_GAMEPLAY_ADMISSION_SECONDS'] = '7200'", "os.environ['TYCHO_GAMEPLAY_ADMISSION_SECONDS'] = '1800'")
    cells=nb['cells'][:3]
    for arm in ('loop','mindset'):
        root=f'/kaggle/working/designer-player-{arm}'
        code=("import os, shutil\nfrom pathlib import Path\n"
              f"arm_root=Path({root!r})\n"
              "if arm_root.exists(): raise RuntimeError('Arm directory exists; no retry into old evidence')\n"
              "arm_root.mkdir()\nshutil.copytree('/kaggle/working/nooa-join',arm_root/'nooa-join')\n"
              f"os.environ['ARC_SOLVER_ARM']={arm!r}\nos.environ['TYCHO_PUBLIC25_WORKING_ROOT']={root!r}\n")
        code+=body.replace('/kaggle/working/',root+'/')
        code+=("subprocess.run([sys.executable,str(arm_root/'nooa-join/assess_candidate.py'),str(output_dir),"
               "'--arm',os.environ['ARC_SOLVER_ARM'],'--output',str(output_dir/'ASSESSMENT.json')],check=True)\n")
        ast.parse(code)
        cells.append(dict(cell_type='code',execution_count=None,metadata={},outputs=[],source=code))
    nb['cells']=cells
    return nb

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true')
    args=parser.parse_args();manifest=verify();nb=build();path=HERE/'probe.ipynb'
    if args.check: assert json.loads(path.read_text())==nb,'Notebook source drift'
    else: path.write_text(json.dumps(nb,indent=2)+'\n')
    OUT.mkdir(parents=True,exist_ok=True);extracted=OUT/'extracted';extracted.mkdir(exist_ok=True)
    code='\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
    payloads=[]
    for node in ast.walk(ast.parse(code)):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='b64decode' and node.args and isinstance(node.args[0],ast.Constant):
            raw=base64.b64decode(node.args[0].value);payloads.append(raw)
            if raw.startswith(b'PK'):
                with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                    for name in archive.namelist():
                        assert not Path(name).is_absolute() and '..' not in Path(name).parts
                        p=extracted/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(archive.read(name))
    for name in ('exact_request_outer_runtime.py','adapter.py','shared_local_solver.py','run_official_agent.py','nooa_public_probe.py'):
        raw=(HERE/name).read_bytes();assert raw in payloads,name;(extracted/name).write_bytes(raw)
    for name,digest in manifest['files'].items():
        assert hashlib.sha256((extracted/name).read_bytes()).hexdigest()==digest,name
    metadata=json.loads((HERE/'kernel-metadata.json').read_text())
    assert metadata['competition_sources']==['arc-prize-2026-arc-agi-3']
    assert metadata['machine_shape']=='NvidiaRtxPro6000'
    launch=dict(kernel=metadata['id'],session_timeout_seconds=10800,arms=['loop','mindset'],
        admission_seconds_per_arm=1800,launcher_timeout_seconds_per_arm=3000,actions_per_arm=30,
        max_output_tokens_per_request=32768,model_context_tokens=262144,
        contract_sha256=manifest['contract_sha256'],promotion=False,launch_ready=True,
        authorization='User explicitly requested notebook construction and Kaggle launch on 2026-09-07.',
        openrouter_qualified=False,automatic_retries=0,
        argv=['kaggle','kernels','push','--path',str(HERE),'--timeout','10800','--accelerator','NvidiaRtxPro6000'],
        files={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in ('probe.ipynb','kernel-metadata.json','prepare_candidate.py','KAGGLE-EXPERIMENT-CARD.md','assess_candidate.py','verify_candidate.py')})
    path=HERE/'LAUNCH.json'
    if args.check: assert json.loads(path.read_text())==launch,'Launch identity drift'
    else:path.write_text(json.dumps(launch,indent=2)+'\n')
    print(json.dumps(launch,indent=2))

if __name__=='__main__':main()
