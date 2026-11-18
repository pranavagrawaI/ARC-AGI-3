"""Read-only frozen/reviewer/receipt binding check. Writes only reviewer summary."""
from pathlib import Path
import ast,base64,hashlib,json
HERE=Path(__file__).resolve().parent;PREP=HERE.parent;ROOT=PREP.parents[1];RUN=PREP/'r10a-cd82-20260916';P=PREP/'r10a-package'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((RUN/'manifest.json').read_text())
assert sha(RUN/'manifest.json')=='9310d597434699539041d342346e630f18ffa295d6f16a4a10205a2229134be9'
assert all(sha(ROOT/n)==v for n,v in manifest['sources'].items())
for f in [HERE/'input-hashes.json',PREP/'independent-repair-review/input-hashes.json',PREP/'freeze-20260916/EVIDENCE-HASHES.json']:
 assert all(sha(ROOT/n)==v for n,v in json.loads(f.read_text()).items())
inv=json.loads((PREP/'R10A-PACKAGE-INVENTORY.json').read_text())['files'];actual={str(p.relative_to(P)):sha(p) for p in P.rglob('*') if p.is_file()};assert inv==actual
nb=json.loads((P/'construction.ipynb').read_text());tree=ast.parse(''.join(nb['cells'][1]['source']));vals={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ['ASSETS','EXPECTED']}
assert set(vals['ASSETS'])==set(actual)-{'construction.ipynb','kernel-metadata.json'}
for n,v in vals['ASSETS'].items():
 b=base64.b64decode(v,validate=True);assert b==(P/n).read_bytes() and hashlib.sha256(b).hexdigest()==vals['EXPECTED'][n]
d=json.loads((PREP/'r10a-verification/package-review/DECISION.json').read_text());assert sha(ROOT/d['report'])==d['report_sha256'];assert d['review_input_hashes_sha256']==sha(PREP/'independent-repair-review/input-hashes.json')
assert d['manifest_sha256']==sha(PREP/'R10A-PACKAGE-INVENTORY.json') and d['notebook_sha256']==manifest['notebook_sha256'] and d['launch_authorized'] is False
sdk=json.loads((PREP/'freeze-20260916/sdk-source-evidence.json').read_text());assert sha(Path(sdk['source_path']))==sdk['source_sha256']
auth=json.loads((RUN/'authorization.json').read_text());assert auth['used'] is False and auth['manifest_sha256']==sha(RUN/'manifest.json')
assert not any((RUN/n).exists() for n in ['hermes-release.json','CONSUMED.json','launch.json'])
summary={'status':'PASS','manifest_sha256':sha(RUN/'manifest.json'),'authorization_sha256':sha(RUN/'authorization.json'),'source_count':len(manifest['sources']),'package_files':len(inv),'embedded_assets':len(vals['ASSETS']),'prior_approval_sha256':sha(PREP/'r10a-verification/package-review/DECISION.json'),'all_input_hashes_match':True,'release_consumption_launch_absent':True,'provider_calls':0}
(HERE/'reconciliation.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
