"""Exercise entry.py/play.py production (non-fixture) branches without /kaggle: expect the fixed competition mount, the gateway default, and override refusal."""
import json,os,subprocess,sys,tempfile,shutil,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
P=ROOT/'deployment/kaggle/submission-20260917-actor-critic-competition-v1'
out={}
def run(name,env_extra,stub_install):
    with tempfile.TemporaryDirectory() as d:
        root=Path(d)
        for n in ('budget.py','entry.py','official-hashes.json'):shutil.copyfile(P/n,root/n)
        (root/'install.py').write_text(stub_install)
        env={k:v for k,v in os.environ.items() if not k.startswith(('ARC_','NOOA_','SUBMISSION_','TYCHO_','KAGGLE_'))}
        env.update(PYTHONDONTWRITEBYTECODE='1',ARC_COMPETITION_STARTED='0',KAGGLE_IS_COMPETITION_RERUN='true',ARC_PROBE_MARK=str(root/'mark.json'),**env_extra)
        code="import sys;sys.path.insert(0,sys.argv[1]);from pathlib import Path;import entry;entry.main(Path(sys.argv[1]))"
        r=subprocess.run([sys.executable,'-B','-c',code,str(root)],env=env,capture_output=True,text=True,timeout=60)
        mark=json.loads((root/'mark.json').read_text()) if (root/'mark.json').exists() else None
        out[name]=dict(returncode=r.returncode,last_stderr=r.stderr.strip().splitlines()[-1] if r.stderr.strip() else '',install_invoked=mark is not None,install_saw_work_deadline=(mark or {}).get('deadline'))
STUB="import json,os;json.dump({'deadline':os.environ.get('ARC_NOTEBOOK_WORK_DEADLINE')},open(os.environ['ARC_PROBE_MARK'],'w'))\n"
# 1: production rerun, no overrides: install.py must run, then the pinned mount is required.
run('production_no_override',{},STUB)
# 2: production rerun with a gateway override present: refused before install.
run('production_gateway_override',{'SUBMISSION_GATEWAY_BASE_URL':'http://127.0.0.1:1'},STUB)
run('production_fixture_key_override',{'ARC_FIXTURE_MODEL_URL':'http://127.0.0.1:1/v1'},STUB)
run('production_nooa_command_override',{'NOOA_AGENT_COMMAND':'x'},STUB)
# 3: the /kaggle/working path is hard-coded in production; probe strings statically.
src=(P/'entry.py').read_text();play=(P/'play.py').read_text()
out['static']=dict(entry_pinned_mount="Path('/kaggle/input/competitions/arc-prize-2026-arc-agi-3')" in src,entry_working="Path('/kaggle/working')" in src,play_gateway_default="'http://gateway:8001'" in play,play_loopback_only="hostname not in ('127.0.0.1','localhost')" in play,play_110_games="len(games) != 110" in play,play_import_shadow_guard="Checkout shadowing detected" in play,play_retries_zero="ARC_LLM_RETRIES" in play)
# 4: play.py gateway override refusal without fixture flag (unit): call run() up to the Arcade construction with a non-loopback override.
print(json.dumps(out,indent=1))
