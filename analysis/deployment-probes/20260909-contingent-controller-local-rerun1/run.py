from pathlib import Path
import json, os, subprocess, time
base=Path('deployment/kaggle/contingent-controller-20260909').resolve()
out=Path('analysis/deployment-probes/20260909-contingent-controller-local-rerun1').resolve()
python='/home/pranav/projects/arc-agi-3-reference/nooa/.venv/bin/python'
env={**os.environ,'PYTHONPATH':str(Path('solvers/tycho_nooa/src').resolve())+':/home/pranav/projects/arc-agi-3-reference/tycho','LITELLM_LOCAL_MODEL_COST_MAP':'True'}
start=time.monotonic()
(out/'STARTED.json').write_text(json.dumps({'time':time.time(),'contract':json.loads((base/'CHASSIS.json').read_text())['contract_sha256']})+'\n')
commands=[[python,'-m','pytest','-q',str(base),'--basetemp',str(out/'pytest/raw'),'--junitxml',str(out/'junit.xml')]]
status=[]
for i,cmd in enumerate(commands):
 with (out/'pytest.log').open('w') as log:
  try:r=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,300-(time.monotonic()-start)));code=r.returncode
  except subprocess.TimeoutExpired:code=124
 status.append({'command':cmd,'exit_code':code})
 if code:break
(out/'EXECUTION.json').write_text(json.dumps({'commands':status,'seconds':time.monotonic()-start},indent=2)+'\n')
print(json.dumps(status))
