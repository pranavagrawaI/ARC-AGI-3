"""Rebuild and qualify locally using this interpreter. No GPU or network calls."""
import hashlib
import json
from pathlib import Path
import os
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
if sys.version_info[:2] != (3,12):
    raise SystemExit('Use CPython 3.12, matching the notebook runtime')
evidence=ROOT/'evidence'
evidence.mkdir(exist_ok=True)
env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'}
checks={
    'build':[str(ROOT/'build.py')],
    'candidate-tests':['-m','unittest','discover','-s',str(ROOT/'tests'),'-v'],
    'trace-replay':[str(ROOT/'tests/replay_traces.py')],
    'package':[str(ROOT/'tests/check_package.py')],
    'lord-receipt':[str(ROOT/'tests/check_donor_receipts.py'),'lordhansolo'],
    'siri-receipt':[str(ROOT/'tests/check_donor_receipts.py'),'sirikilohit'],
}
for name,args in checks.items():
    with (evidence/(name+'.log')).open('w') as log:
        run=subprocess.run([sys.executable,*args],stdout=log,stderr=subprocess.STDOUT,env=env)
    if run.returncode:
        raise SystemExit(f'{name} FAILED; see {evidence/(name+".log")}')
    print(name,'PASS',flush=True)
parent=ROOT.parent/'m2_basics'
parent_hashes=json.loads((evidence/'parent-hashes.json').read_text())
for rel,sha in parent_hashes.items():
    assert hashlib.sha256((parent/rel).read_bytes()).hexdigest()==sha,rel
files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
       for p in sorted(ROOT.rglob('*')) if p.is_file() and '__pycache__' not in str(p)
       and p!=evidence/'verification.json'}
(evidence/'verification.json').write_text(json.dumps({'python':sys.version,'checks':list(checks),
    'parent_unchanged':True,'files':files,'scope':'CPU only; no model or GPU execution'},indent=2)+'\n')
print('Parent M2 Basics unchanged; artifact hashes recorded')
