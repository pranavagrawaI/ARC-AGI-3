"""Hash every file under the package and the manifest-bound run files; print JSON."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
P=ROOT/'deployment/kaggle/submission-20260917-actor-critic-competition-v1'
R=ROOT/'analysis/competition-submission-20260917-actor-critic-v1'
m=json.loads((R/'manifest.json').read_text())
out={}
for p in sorted(P.rglob('*')):
    if p.is_file() or p.is_symlink(): out[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
for n in m['sources']:
    out[n]=hashlib.sha256((ROOT/n).read_bytes()).hexdigest()
out['__manifest__']=hashlib.sha256((R/'manifest.json').read_text().encode()).hexdigest()
out['__pycache_dirs__']=[str(p.relative_to(ROOT)) for p in P.rglob('__pycache__')]
print(json.dumps(out,indent=1,sort_keys=True))
