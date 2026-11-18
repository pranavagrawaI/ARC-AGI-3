"""Reproduce offline checks; never invokes Kaggle or a live provider."""
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
repo=HERE.parents[1]
env=dict(os.environ,PYTHONPATH=str(HERE))
proc=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(HERE/'tests'),'-v'],cwd=repo,env=env,capture_output=True,text=True,timeout=180)
(HERE/'LOCAL-TESTS.log').write_text(proc.stdout+proc.stderr)
blocked=subprocess.run([sys.executable,str(HERE/'run.py'),'--execute'],cwd=repo,env=env,capture_output=True,text=True,timeout=10)
prior=HERE.parent/'model-construction-real-prefix-v4-20260913'
started=json.loads((prior/'STARTED.json').read_text())
prior_checks={n:hashlib.sha256((prior/n).read_bytes()).hexdigest()==digest for n,digest in started['artifact_sha256'].items()}
count=re.search(r'Ran (\d+) tests',proc.stderr)
summary={'passed':proc.returncode==0 and all(prior_checks.values()) and blocked.returncode!=0 and 'Kaggle isolation is unqualified' in blocked.stderr,
         'tests_run':int(count.group(1)) if count else None,'test_exit_code':proc.returncode,
         'launch_guard_verified':blocked.returncode!=0 and 'Kaggle isolation is unqualified' in blocked.stderr,
         'v4_frozen_sources_unchanged':prior_checks,
         'subject_sha256':{str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(HERE.rglob('*.py'))},
         'scope':'Offline captured-response and mock-HTTP machinery checks only. No model construction episode or GPU run.'}
(HERE/'LOCAL-VERIFICATION.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
raise SystemExit(0 if summary['passed'] else 1)
