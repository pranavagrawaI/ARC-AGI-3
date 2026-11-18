"""Competition entry: commit is cheap; only rerun starts the solver."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys,time,zipfile
import budget

def validate_output(path, games):
    import pandas as pd
    frame=pd.read_parquet(path)
    if list(frame.columns) != ['row_id','game_id','end_of_game','score'] or frame.empty:
        raise RuntimeError('Invalid gateway parquet schema')
    if frame.isna().any().any() or frame.row_id.duplicated().any():
        raise RuntimeError('Null/duplicate gateway rows')
    if set(frame.game_id.astype(str)) != set(games):
        raise RuntimeError('Gateway parquet game set incomplete')
    if not set(frame.end_of_game.unique()) <= {True,False}:
        raise RuntimeError('Invalid end_of_game values')
    if set(frame.loc[frame.end_of_game.astype(bool),'game_id'].astype(str)) != set(games):
        raise RuntimeError('Gateway output lacks terminal row for every game')
    return len(frame)

def main(root):
    fixture=os.environ.get('ARC_LOCAL_FIXTURE')=='1'
    work=root/'working' if fixture else Path('/kaggle/working')
    work.mkdir(exist_ok=True,parents=True)
    rerun=os.environ.get('KAGGLE_IS_COMPETITION_RERUN','').lower() in ('true','1')
    if not rerun:
        import pandas as pd
        pd.DataFrame([['1_0','1',True,1]],columns=['row_id','game_id','end_of_game','score']).to_parquet(work/'submission.parquet',index=False)
        return
    budget.validate()
    started=float(os.environ['ARC_COMPETITION_STARTED'])
    deadline=started+budget.PROVIDER-budget.UNCERTAINTY-budget.FINAL_OUTPUT-budget.OUTER_CLEANUP
    os.environ['ARC_NOTEBOOK_WORK_DEADLINE']=str(deadline)
    if fixture:
        competition=Path(os.environ['ARC_FIXTURE_COMPETITION'])
    else:
        if any(k.startswith('ARC_FIXTURE_') or k in ('SUBMISSION_GATEWAY_BASE_URL','NOOA_AGENT_COMMAND') for k in os.environ):
            raise RuntimeError('Fixture override in production')
        competition=Path('/kaggle/input/competitions/arc-prize-2026-arc-agi-3')
        subprocess.run([sys.executable,str(root/'install.py')],check=True,timeout=budget.BOOTSTRAP)
    source=competition/'ARC-AGI-3-Agents'
    hashes=json.loads((root/'official-hashes.json').read_text())
    for name,digest in hashes.items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Official framework drift: '+name)
    framework=work/'ARC-AGI-3-Agents'
    framework.mkdir(exist_ok=False)
    for name in hashes:
        if name.endswith('.py'):
            dst=framework/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/name,dst)
    (framework/'runtime').mkdir()
    (framework/'runtime/__init__.py').write_text('')
    for src,dst in [('my_agent.py','agents/templates/my_agent.py'),('shared_local_solver.py','runtime/local_solver.py'),('wrapper_local_solver.py','local_solver.py')]:
        shutil.copyfile(root/src,framework/dst)
    (framework/'agents/__init__.py').write_text('from .agent import Agent, Playback\nfrom .swarm import Swarm\n')
    join=work/'nooa-join'
    if fixture:
        join.mkdir();zipfile.ZipFile(root/'nooa-join.zip').extractall(join)
    os.environ.update({'NOOA_JOIN_ROOT':str(join),'SUBMISSION_FRAMEWORK':str(framework),
        'SUBMISSION_ENVIRONMENTS':str(competition/'environment_files'),
        'ARC_STALL_REVIEW':'0','ARC_DISCOVERY_ARM':'a','ARC_SOLVER_ARM':'loop','ARC_CONTINGENT_ARM':'on',
        'ARC_DISABLE_DOTENV':'1','ARC_LLM_CONTEXT_WINDOW':'262144','ARC_LLM_MAX_TOKENS':'8192',
        'ARC_LLM_RETRIES':'0','ARC_LLM_TIMEOUT_SECONDS':str(budget.REQUEST),'ARC_LLM_USE_RESPONSES':'0',
        'ARC_LLM_EXTRA_BODY':'{"chat_template_kwargs":{"enable_thinking":false},"reasoning":{"enabled":false}}',
        'ARC_COMPETITION_ACTOR_CAP':str(budget.ACTOR_CALLS),'ARC_COMPETITION_CRITIC_CAP':str(budget.CRITIC_CALLS),
        'SUBMISSION_ACTION_LIMIT':'200','PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1'})
    output=work/'qualification';output.mkdir(exist_ok=False)
    # Validate the worst-case startup envelope before any model process starts.
    remaining=min(budget.GAMEPLAY,int(deadline-time.monotonic()-budget.STARTUP_WITH_WARMUP-30))
    budget.admission(remaining,budget.GAMES)
    if fixture:
        command=[sys.executable,str(root/'submission_play.py'),'--base-url',os.environ['ARC_FIXTURE_MODEL_URL'],
            '--model','qwen/qwen3.8-27b','--workspace-root',str(output/'workspaces'),
            '--output',str(output/'launcher-result.json'),'--timeout-s',str(remaining)]
    else:
        command=[sys.executable,str(root/'official_launcher_runtime.py'),'--output-dir',str(output),
            '--manifest',str(root/'PROTOTYPE-qwen3_8_fp8-manifest.json'),'--input-root','/kaggle/input',
            '--model-source','jakobbrggen/qwen3-8-27b-fp8-hf-snapshot/versions/1',
            '--runtime-provenance','/tmp/sglang-runtime-provenance.json','--startup-timeout-s',str(budget.STARTUP_WITH_WARMUP),
            '--request-timeout-s','180','--launcher-timeout-s',str(remaining),
            '--attention-backend','triton','--sampling-backend','pytorch','--profile-imports','--r8-l3b',
            '--official-launcher-probe',str(root/'submission_play.py'),'--launcher-python','/tmp/solver-venv/bin/python']
    (output/'command.json').write_text(json.dumps(command))
    subprocess.run(command,check=True,timeout=max(1,deadline-time.monotonic()+budget.OUTER_CLEANUP))
    progress=[json.loads(s) for s in (output/'game-progress.jsonl').read_text().splitlines()]
    validate_output(work/'submission.parquet',[r['game_id'] for r in progress])

if __name__=='__main__':main(Path(__file__).resolve().parent)
