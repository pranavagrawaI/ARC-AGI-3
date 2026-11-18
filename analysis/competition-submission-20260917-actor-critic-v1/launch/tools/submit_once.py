"""Exactly one competition submission of the exact pushed version. O_EXCL marker consumed before the provider call."""
import json,os,subprocess,sys,time
from pathlib import Path
RUN=Path(__file__).resolve().parents[2];CLI='/home/pranav/.local/bin/kaggle';VERSION=sys.argv[1]
COMP='arc-prize-2026-arc-agi-3';TARGET='pranavagrawai/arc3-ac-competition-20260917-v1'
CMD=[CLI,'competitions','submit',COMP,'-k',TARGET,'-v',VERSION,'-f','submission.parquet','-m','Actor-critic solver v1']
auth=json.loads((RUN/'SUBMISSION-AUTHORIZATION.json').read_text())
assert auth['used'] is False and auth['max_submissions']==1 and auth['bound_version']==int(VERSION) and auth['competition']==COMP and auth['target']==TARGET,'submission authorization not bound to this version'
marker=RUN/'SUBMISSION-CONSUMED.json'
fd=os.open(marker,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
with os.fdopen(fd,'w') as f:json.dump(dict(target=TARGET,version=int(VERSION),competition=COMP,command=CMD,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())),f);f.flush();os.fsync(f.fileno())
t0=time.time()
try:r=subprocess.run(CMD,text=True,capture_output=True,timeout=300);rec=dict(command=CMD,started_unix=t0,finished_unix=time.time(),returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
except subprocess.TimeoutExpired as e:rec=dict(command=CMD,started_unix=t0,finished_unix=time.time(),returncode=None,timeout=True,stdout=str(e.stdout),stderr=str(e.stderr))
with (RUN/'submission-receipt.json').open('x') as f:json.dump(rec,f,indent=2)
print(json.dumps(rec,indent=1))
