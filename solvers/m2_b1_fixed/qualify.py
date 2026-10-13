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
GATE_PYTHON=ROOT.parent/'tycho_nooa/.venv/bin/python'  # has requests/PIL/numpy for the real ToolAgent
for name,args in checks.items():
    with (evidence/(name+'.log')).open('w') as log:
        run=subprocess.run([sys.executable,*args],stdout=log,stderr=subprocess.STDOUT,env=env)
    if run.returncode:
        raise SystemExit(f'{name} FAILED; see {evidence/(name+".log")}')
    print(name,'PASS',flush=True)
with (evidence/'evidence-gate.log').open('w') as log:
    run=subprocess.run([str(GATE_PYTHON),'-B',str(ROOT/'tests/gate_evidence.py')],stdout=log,stderr=subprocess.STDOUT,env=env)
if run.returncode:
    raise SystemExit('evidence-gate FAILED; see '+str(evidence/'evidence-gate.log'))
print('evidence-gate PASS',flush=True)
with (evidence/'b1-gate.log').open('w') as log:
    run=subprocess.run([str(GATE_PYTHON),'-B',str(ROOT/'tests/gate_b1.py')],stdout=log,stderr=subprocess.STDOUT,env=env)
if run.returncode:
    raise SystemExit('b1-gate FAILED; see '+str(evidence/'b1-gate.log'))
print('b1-gate PASS',flush=True)
with (evidence/'pin-replay.log').open('w') as log:
    run=subprocess.run([str(GATE_PYTHON),'-B',str(ROOT/'tests/replay_pin.py')],stdout=log,stderr=subprocess.STDOUT,env=env)
if run.returncode:
    raise SystemExit('pin-replay FAILED; see '+str(evidence/'pin-replay.log'))
print('pin-replay PASS',flush=True)
for name, script in (('bugfix-gate', 'gate_bugfix.py'), ('parent-comparison', 'compare_parent.py')):
    with (evidence/(name+'.log')).open('w') as log:
        run=subprocess.run([str(GATE_PYTHON),'-B',str(ROOT/'tests'/script)],stdout=log,stderr=subprocess.STDOUT,env=env)
    if run.returncode:
        raise SystemExit(name+' FAILED; see '+str(evidence/(name+'.log')))
    print(name,'PASS',flush=True)
parent=ROOT.parent/'m2_b1'
parent_hashes=json.loads((evidence/'parent-hashes.json').read_text())
for rel,sha in parent_hashes.items():
    assert hashlib.sha256((parent/rel).read_bytes()).hexdigest()==sha,rel
files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
       for p in sorted(ROOT.rglob('*')) if p.is_file() and '__pycache__' not in str(p)
       and p!=evidence/'verification.json'}
(evidence/'verification.json').write_text(json.dumps({'python':sys.version,'checks':[*checks,'evidence-gate','b1-gate','pin-replay','bugfix-gate','parent-comparison'],
    'parent_unchanged':True,'files':files,'scope':'CPU only; no model or GPU execution'},indent=2)+'\n')
print('Parent M2 B1 unchanged; artifact hashes recorded')
