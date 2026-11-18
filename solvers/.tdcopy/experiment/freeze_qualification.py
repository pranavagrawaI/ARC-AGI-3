"""Prepare immutable CPU setup and one-hour GPU qualification artifacts."""
from __future__ import annotations
import ast
import argparse
import hashlib
import json
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def cell(source):return {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],
                         'source':source.splitlines(True)}


def main():
    import nbformat
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision',choices=['q5','q6'],default='q5')
    args=parser.parse_args()
    manifest=json.loads((HERE/'manifest.json').read_text())
    if args.revision=='q6':
        previous=json.loads((HERE/'experiment/gpu-q5/manifest.json').read_text())
        assert set(previous['source_files'])==set(manifest['source_files'])
        changed={name for name,digest in manifest['source_files'].items()
                 if digest!=previous['source_files'][name]}
        assert changed=={'hybrid/runner.py'},changed
    original=json.loads((HERE/'kernel/tycho-dfranzen.ipynb').read_text())
    assert sha(HERE/'kernel/tycho-dfranzen.ipynb')==manifest['notebook_sha256']
    watchdog=(HERE/'experiment/watchdog.py').read_text()
    for run_id,cpu in [('cpu-'+args.revision,True),('gpu-'+args.revision,False)]:
        directory=HERE/'experiment'/run_id
        directory.mkdir(exist_ok=False)
        (directory/'kernel').mkdir()
        for name in ['manifest.json','canonical-candidate.patch','build.py','verify.py']:
            shutil.copy2(HERE/name,directory/name)
        shutil.copytree(HERE/'src',directory/'src',ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copytree(HERE/'tests',directory/'tests',ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copytree(HERE/'source-bundle',directory/'source-bundle')
        for name in ['collect_run.py','collect_outputs.py','audit_qualification.py','watchdog.py']:
            shutil.copy2(HERE/'experiment'/name,directory/name)
        notebook=json.loads(json.dumps(original))
        if cpu:
            notebook['cells']=[c for i,c in enumerate(notebook['cells'][:12]) if i!=6]
            preflight=f'''import json,os,time,subprocess
from pathlib import Path
_qualification_root=Path('/kaggle/working')
_qualification_root.mkdir(exist_ok=True)
(_qualification_root/'qualification-start.json').write_text(json.dumps({{'run_id':{run_id!r},'started_epoch':time.time(),'gpu_count':0,'claim':'CPU installation/execution rehearsal only'}}))
'''
            fixture='''# No live LLM calls: explicit scripted CPU transport fixture.
import asyncio,os,pickle
from pathlib import Path
_fixture_code = """import asyncio,os,sys,json,pickle
sys.path.insert(0,'/kaggle/working/tycho-dfranzen-runtime')
from pathlib import Path
import arcengine
from taaf.benchmark import Benchmark
from taaf.game_examples import ExampleGame
from taaf.deploy_inline import InlineTarget
from inference.framework.solver import HarnessSolver
from hybrid.runner import main
os.environ['TYCHO_MODE']='no_world_model'
os.environ['TYCHO_NOOA_TREATMENT']='control'
os.environ['HYBRID_RUN_MODE']='public25'
os.environ['ARC3_MAX_ACTIVE_STREAMS']='0'
os.environ['ARC3_PRIORITY_TAIL_FADE']='0'
os.environ['TYCHO_NOOA_SCRIPTED_TOOL_CALLS']=json.dumps([
 {'tool':'run_python','arguments':{'code':"print('CPU_COMPONENTS',len(dfr.segment([[0,1],[0,1]])['nodes']))"}},
 {'tool':'take_actions','arguments':{'actions':[{'action':'UP'}],'rationale':'CPU fixture'}},
 {'tool':'write_file','arguments':{'path':'notes/level_0_insights.md','content':'Scripted CPU fixture.'}}])
root=Path('/kaggle/working')
bm=Benchmark(games=[ExampleGame(target_sequence_per_level=[[arcengine.GameAction.ACTION1]],label='cpu-fixture')],
 job_dir=root,solver=HarnessSolver(max_runtime_s_per_game=60,max_actions_per_game=1,analyzer_timeout=30))
target=InlineTarget();target.actual_run_as_submission=False
(root/'cpu-bm.pkl').write_bytes(pickle.dumps(bm));(root/'cpu-target.pkl').write_bytes(pickle.dumps(target))
sys.argv=['runner',str(root/'cpu-bm.pkl'),str(root/'cpu-target.pkl')]
main()
saved=json.loads((root/'tycho-dfranzen/benchmark.json').read_text())
assert saved['game_runs'][0]['state']=='won',saved
traces=[p.read_text() for p in (root/'tycho-dfranzen/hybrid-games').glob('*/tool-trace.jsonl')]
assert any('CPU_COMPONENTS 2' in t for t in traces)
print('CPU hardened worker, native donor computation, action and Scribe fixture passed')
"""
_path=Path('/tmp/hybrid-cpu-fixture.py');_path.write_text(_fixture_code)
subprocess.run([str(HYBRID_PYTHON),str(_path)],check=True,timeout=120)
(_qualification_root/'qualification-complete.json').write_text(json.dumps({'cpu_fixture_passed':True}))
'''
            if args.revision=='q6':
                code=(HERE/'experiment/cpu_gameapi_fixture.py').read_text()
                shutil.copy2(HERE/'experiment/cpu_gameapi_fixture.py',directory/'cpu_gameapi_fixture.py')
                fixture=f'''# Actual GameAPI rehearsal; explicit scripted client, no endpoint requests.
_fixture_path=Path('/tmp/hybrid-cpu-gameapi-fixture.py')
_fixture_path.write_text({code!r})
subprocess.run([str(HYBRID_PYTHON),str(_fixture_path),'--runtime',str(HYBRID_ROOT),
 '--environments','/kaggle/input/competitions/arc-prize-2026-arc-agi-3/environment_files',
 '--output',str(_qualification_root)],check=True,timeout=240)
'''
            notebook['cells'].append(cell(fixture))
        else:
            preflight=f'''import json,os,sys,time,subprocess
from pathlib import Path
_qualification_root=Path('/kaggle/working');_qualification_root.mkdir(exist_ok=True)
_qualification_started_monotonic=time.monotonic()
_qualification_started_epoch=time.time()
_watchdog_path=Path('/tmp/hybrid-qualification-watchdog.py')
_watchdog_path.write_text({watchdog!r})
_watchdog=subprocess.Popen([sys.executable,str(_watchdog_path),'--parent',str(os.getpid()),
 '--root',str(_qualification_root),'--started',str(_qualification_started_monotonic),
 '--soft','3300','--hard','3600','--force-margin','10'],start_new_session=True)
os.environ['HYBRID_RUN_MODE']='small'
_gpu_inventory=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid','--format=csv,noheader'],text=True).strip().splitlines()
(_qualification_root/'qualification-start.json').write_text(json.dumps({{'run_id':{run_id!r},'started_epoch':_qualification_started_epoch,
 'gpu_inventory':_gpu_inventory,'hard_timeout_seconds':3600,'soft_stop_seconds':3300,
 'per_game_seconds':2400,'action_cap':None,'claim':'runtime/depth qualification; no comparative efficacy'}}))
assert len(_gpu_inventory)==1 and 'RTX PRO 6000' in _gpu_inventory[0], 'exactly one donor GPU required'
'''
            source=''.join(notebook['cells'][21]['source'])
            before="if soft_end: _command.extend(['--soft-end',soft_end.isoformat()])"
            assert source.count(before)==1
            source=source.replace(before,"soft_end=datetime.fromtimestamp(_qualification_started_epoch+3300)\n"+before)
            notebook['cells'][21]['source']=source.splitlines(True)
            notebook['cells'].append(cell("(_qualification_root/'qualification-complete.json').write_text(json.dumps({'notebook_completed':True}))\n"))
        notebook['cells'].insert(0,cell(preflight))
        metadata=json.loads((HERE/'kernel/kernel-metadata.json').read_text())
        metadata.update(id='pranavagrawai/tycho-dfranzen-'+run_id,title='Tycho dfranzen '+run_id)
        if cpu:
            metadata['enable_gpu']=False;metadata.pop('machine_shape',None)
        path=directory/'kernel/tycho-dfranzen.ipynb'
        path.write_text(json.dumps(notebook,indent=1)+'\n')
        meta_path=directory/'kernel/kernel-metadata.json';meta_path.write_text(json.dumps(metadata,indent=2)+'\n')
        nbformat.validate(nbformat.from_dict(notebook))
        assert path.stat().st_size<1_000_000
        for i,c in enumerate(notebook['cells']):
            source=''.join(c['source'])
            if c['cell_type']=='code' and not any(line.startswith(('!','%')) for line in source.splitlines()):
                compile(source,str(i),'exec',flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        identity={'run_id':run_id,'notebook_sha256':sha(path),'metadata_sha256':sha(meta_path),
                  'parent_notebook_sha256':manifest['notebook_sha256'],
                  'source_manifest_sha256':sha(directory/'manifest.json'),
                  'source_payload_sha256':manifest['source_payload_sha256'],
                  'gpu_count':0 if cpu else 1,'hard_timeout_seconds':600 if cpu else 3600}
        (directory/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
        card=f'''# {run_id}
Question: {'Do the exact private source and dependency attachments install offline and execute the hardened scripted CPU path?' if cpu else 'Does v3 execute deeper real-model games correctly, with construction controlled by the Actor?'}
Posture: execution qualification; no comparative efficacy claim.
Contrast: none.
Prediction: {'Hash-bound extraction, offline installation and hardened scripted fixture pass.' if cpu else 'Three games progress beyond the shallow smoke; request/action/role receipts reconcile. Builder requests receive invocation-time allocation when time remains.'}
Measure: qualification-start.json, qualification-complete.json, tycho-dfranzen/candidate-config.json, benchmark.json, evidence-index.json, per-game compute/host-actions/tool/LLM/exit receipts and provider checkpoints. {('Actual ar25/ls20/sb26 GameAPI objects must each acknowledge one legal scripted action and report CPU_COMPONENTS 2. Three sequential one-game rehearsals are aggregated without altering game runs; raw benchmark/config inputs survive. No score/model-capability inference.' if args.revision=='q6' else 'Scripted fixture must win and report CPU_COMPONENTS 2; no GPU/model-capability inference.') if cpu else 'Count offered/requested/admitted/completed Builder calls separately; non-use is allowed and leaves conditional efficacy unidentified.'}
Cap: {'zero GPUs; API session timeout 600 seconds; three sequential scripted fixture games, 60 seconds/game and 1 action each, 240-second fixture subprocess limit; zero paid endpoint calls.' if cpu and args.revision=='q6' else 'zero GPUs; API session timeout 600 seconds; scripted fixture 60 seconds/game and 1 action; zero paid endpoint calls.' if cpu else 'one RTX PRO 6000; API session timeout 3600 seconds including startup; independent watchdog forces exit at 3590 seconds; soft stop 3300 seconds; 3 games ar25/ls20/sb26, one pass, concurrency 3, 2400 seconds/game, no action ceiling, 12288 output tokens/call; at most 1 GPU-hour reserved.'} Total campaign cap 10 GPU-hours; no submission/public publication; no automatic retry.
Stop: end after qualification; preserve partial outputs on integrity/runtime/install failure. A crashed game prevents comparative interpretation and does not become a capability zero.
Invalid if: source/config/hardware/game identity differs, hashes fail, illegal/duplicate/unacknowledged actions, action after cutoff, missing required receipts, failed isolation/installation, or unintended suppression. Expected resource-cutoff give-up is censored; interrupted provider usage is unknown without receipt. No provider cancellation claim.
Result map: {f'Pass -> permit frozen gpu-{args.revision} launch; failure -> preserve artifacts and diagnose, no GPU launch.' if cpu else 'Valid execution -> prepare a fresh matched pair after auditing traces and score headroom; unused construction -> retain non-use, no inert ablation; admitted construction -> inspect predictive support/repair before choosing a fresh mechanism contrast; failure -> preserve and diagnose before any distinct bounded retry; sparse evidence -> inconclusive.'}
Outputs: {directory.relative_to(HERE).as_posix()}/output and frozen source/card/identity/launch/status/collection receipts.
Authority: user authorized campaign and instructed proceeding on 2026-10-04; this card grants no additional authority.
Notebook SHA256: {identity['notebook_sha256']}
Metadata SHA256: {identity['metadata_sha256']}
Source manifest SHA256: {identity['source_manifest_sha256']}
Source payload SHA256: {identity['source_payload_sha256']}
Retry identity: {args.revision}; q6 is explicitly authorized after q5 failed before gameplay. Frozen q5 evidence is retained. Only the eager game-name fallback changed in model runtime since q5; source-v4 is a distinct hash-bound dataset. Qualification does not compare reasoning efficacy.
'''
        (directory/'EXPERIMENT-CARD.md').write_text(card)
        print(json.dumps(identity))


if __name__=='__main__':main()
