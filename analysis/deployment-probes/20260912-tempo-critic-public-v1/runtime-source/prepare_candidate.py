"""Build/check the two-arm reasoning-off contingent-controller notebook."""
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
ARMS=('off','on')
# One source for the budget. v1's off arm committed 10 actions in 1,046s of
# gameplay (104.6s/action), so 20 actions project to ~2,091s: an 1,800s
# admission would stop the arm at wall_budget around action 17 and waste the
# bump. Admission carries headroom over that projection, and the launcher
# timeout must additionally cover server startup (~400s) and teardown.
ACTIONS_PER_ARM='20'
ADMISSION_SECONDS='3000'
LAUNCHER_TIMEOUT_SECONDS=3600
OUT=HERE.parents[2]/'analysis/deployment-probes/20260909-contingent-controller-package'

def build():
    off='{"chat_template_kwargs":{"enable_thinking":false},"reasoning":{"enabled":false}}'
    nb=builder.build(max_actions=ACTIONS_PER_ARM,worker_timeout_s=900,launcher_timeout_s=LAUNCHER_TIMEOUT_SECONDS,
        max_tokens='16384',llm_timeout_s=300,request_timeout_s=180,extra_body=off,
        heading='# Contingent controller contrast on Kaggle\n\noff=the contingent interface is absent, on=it is available. Discovery arm A, Actor reasoning off, fixed FP8 baseline and MTP off in both arms. Only ARC_CONTINGENT_ARM differs. Public ls20 only; no submission or automatic promotion.')
    builder.validate(nb,max_actions=ACTIONS_PER_ARM,launcher_timeout_s=LAUNCHER_TIMEOUT_SECONDS,extra_body=off)
    body=''.join(nb['cells'][3]['source']).split('submission = pd.DataFrame(')[0]
    body=body.replace("os.environ['TYCHO_GAMEPLAY_ADMISSION_SECONDS'] = '7200'", f"os.environ['TYCHO_GAMEPLAY_ADMISSION_SECONDS'] = {ADMISSION_SECONDS!r}")
    cells=nb['cells'][:3]
    # The contingent interface is the only factor. Discovery arm A, solver arm
    # loop, budgets, gates and model settings are identical in both arms; the
    # off arm hides the two tools, their instructions and the dispatcher path.
    for arm in ARMS:
        root=f'/kaggle/working/contingent-{arm}'
        code=("import os, shutil\nfrom pathlib import Path\n"
              f"arm_root=Path({root!r})\n"
              "if arm_root.exists(): raise RuntimeError('Arm directory exists; no retry into old evidence')\n"
              "arm_root.mkdir()\nshutil.copytree('/kaggle/working/nooa-join',arm_root/'nooa-join')\n"
              "os.environ['ARC_SOLVER_ARM']='loop'\n"
              "os.environ['ARC_DISCOVERY_ARM']='a'\n"
              f"os.environ['ARC_CONTINGENT_ARM']={arm!r}\nos.environ['TYCHO_PUBLIC25_WORKING_ROOT']={root!r}\n")
        code+=body.replace('/kaggle/working/',root+'/')
        # The assessor is told the arm and budgets this cell configured, so a
        # dropped factor or a wrong limit fails the arm instead of passing quietly.
        code+=("subprocess.run([sys.executable,str(arm_root/'nooa-join/assess_candidate.py'),str(output_dir),"
               "'--arm',os.environ['ARC_SOLVER_ARM'],'--reasoning','off',"
               "'--contingent-arm',os.environ['ARC_CONTINGENT_ARM'],"
               "'--action-limit',os.environ['TYCHO_KAGGLE_MAX_ACTIONS'],"
               "'--admission-seconds',os.environ['TYCHO_GAMEPLAY_ADMISSION_SECONDS'],"
               "'--output',str(output_dir/'ASSESSMENT.json')],check=True)\n")
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
    launch=dict(kernel=metadata['id'],session_timeout_seconds=10800,arms=list(ARMS),
        factor='ARC_CONTINGENT_ARM',discovery_arm='a',
        admission_seconds_per_arm=int(ADMISSION_SECONDS),launcher_timeout_seconds_per_arm=LAUNCHER_TIMEOUT_SECONDS,
        actions_per_arm=int(ACTIONS_PER_ARM),
        max_output_tokens_per_request=16384,model_context_tokens=262144,
        contract_sha256=manifest['contract_sha256'],promotion=False,launch_ready=True,
        authorization='Bounded two-arm contingent-controller contrast on public ls20. No submission, retry or automatic promotion.',
        openrouter_qualified=False,automatic_retries=0,
        argv=['kaggle','kernels','push','--path',str(HERE),'--timeout','10800','--accelerator','NvidiaRtxPro6000'],
        files={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in ('probe.ipynb','kernel-metadata.json','prepare_candidate.py','KAGGLE-EXPERIMENT-CARD.md','assess_candidate.py','verify_candidate.py')})
    path=HERE/'LAUNCH.json'
    if args.check: assert json.loads(path.read_text())==launch,'Launch identity drift'
    else:path.write_text(json.dumps(launch,indent=2)+'\n')
    print(json.dumps(launch,indent=2))

if __name__=='__main__':main()
