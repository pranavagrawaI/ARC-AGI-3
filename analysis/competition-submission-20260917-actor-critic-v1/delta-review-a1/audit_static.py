import ast,base64,difflib,hashlib,importlib.util,io,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];R=Path(__file__).resolve().parents[1];O=Path(__file__).resolve().parent;P=ROOT/'deployment/kaggle/submission-20260917-actor-critic-competition-v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
b=json.loads((R/'PRE-REPAIR-HASHES.json').read_text())
assert len(b['independent_review'])==6536
for n,h in b['independent_review'].items():assert sha(ROOT/n)==h,n
before=R/'checks/repair-a1/before-package'
for n,h in b['package'].items():
 q=ROOT/n
 rel=q.relative_to(P) if q.is_relative_to(P) else Path(n)
 assert sha(before/rel)==h,n
changed=[str(p.relative_to(P)) for p in P.rglob('*') if p.is_file() and sha(p)!=sha(before/p.relative_to(P))]
assert set(changed)=={'runtime/chassis/solver_agent.py','runtime/headless.py','runtime/CHASSIS.json','ASSETS.json','submission.ipynb'},changed
with (O/'package-source.diff').open('w') as f:
 for n in ['runtime/chassis/solver_agent.py','runtime/headless.py']:
  f.writelines(difflib.unified_diff((before/n).read_text().splitlines(True),(P/n).read_text().splitlines(True),fromfile='before/'+n,tofile='current/'+n))
new=(P/'runtime/chassis/solver_agent.py').read_text();old=(before/'runtime/chassis/solver_agent.py').read_text()
guard='        if os.environ.get("ARC_GAME_REQUEST_LEDGER"):\n            raise RuntimeError("deep investigation is disabled under competition request accounting")\n'
assert new.replace(guard,'',1)==old and new.count(guard)==1
prompt=(P/'runtime/headless.py').read_text();prior=(before/'runtime/headless.py').read_text()
assert 'Use await self.deep_investigate' not in prompt
m=json.loads((R/'manifest.json').read_text());assert len(m['sources'])==85
for n,h in m['sources'].items():assert sha(ROOT/n)==h,n
assert not any(p.is_symlink() or p.name=='__pycache__' or p.suffix=='.pyc' for p in P.rglob('*'))
assert len([p for p in P.rglob('*') if p.is_file()])==72
spec=importlib.util.spec_from_file_location('gate_audit',R/'launch_control.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g);g.reconcile(str(R))
nb=json.loads((P/'submission.ipynb').read_text());payloads=[]
for c in nb['cells']:
 if c['cell_type']=='code':
  for n in ast.walk(ast.parse(''.join(c['source']))):
   if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='b64decode':payloads.append(base64.b64decode(n.args[0].value))
assert len(payloads)==13
for p in payloads:assert b'-----BEGIN PRIVATE KEY' not in p
z=next(zipfile.ZipFile(io.BytesIO(p)) for p in payloads if p.startswith(b'PK'))
ch=json.loads((P/'runtime/CHASSIS.json').read_text());assert json.loads(z.read('CHASSIS.json'))==ch
for n,h in ch['files'].items():assert hashlib.sha256(z.read(n)).hexdigest()==h
contents={k:ch[k] for k in ('files','sources')};assert hashlib.sha256(json.dumps(contents,sort_keys=True).encode()).hexdigest()==ch['contract_sha256']
# Recover the old defect from exact pre-repair materialized notebook, server requests, ledger and IPC.
d=R/'checks/repair-a1/deep-before';oldnb=json.loads((before/'submission.ipynb').read_text());assert (d/'execute-notebook.py').read_text()=='\n'.join(''.join(c['source']) for c in oldnb['cells'] if c['cell_type']=='code')
req=[json.loads(l) for l in (d/'model-requests.jsonl').read_text().splitlines()];q=d/'notebook/working/qualification';ledger=[json.loads(l) for p in q.glob('request-budget-*.jsonl') for l in p.read_text().splitlines()];deep=[json.loads(l) for p in q.rglob('deep_investigations.jsonl') for l in p.read_text().splitlines()]
assert len(req)==14 and len(ledger)==13 and len(deep)==1
assert sum(not r['request'].get('tools') and r['request'].get('max_tokens')==32768 for r in req)==1
assert "'expected_launcher_solver_requests': 13" in (d/'RECONCILE-ERROR.txt').read_text() and "'observed_launcher_solver_requests': 14" in (d/'RECONCILE-ERROR.txt').read_text()
assert json.loads((d/'RESULT.json').read_text())['status']=='reconciliation_failed'
assets=json.loads((before/'ASSETS.json').read_text());assert all(sha(d/'notebook'/n)==h for n,h in assets.items())
result=dict(protected_prior_review_files=6536,manifest_sources=85,package_files=72,changed_package_files=sorted(changed),payloads=13,chassis_entries=len(z.namelist()),contract=ch['contract_sha256'],bindings=g.bindings(m),metadata=m['metadata'],prior_defect=dict(server=14,ledger=13,deep=1,status='reconciliation_failed'),final_review_initially_absent=not (R/'final-review.json').exists())
(O/'static-results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
