"""Create the immutable final manifest after all evidence exists."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
PACKAGE=REPO/'deployment/kaggle'/HERE.name

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
files={}
for p in sorted(HERE.rglob('*')):
    if not p.is_file() or '__pycache__' in p.parts or p.name=='FINAL-MANIFEST.json': continue
    files[str(p.relative_to(REPO))]=sha(p)
for p in sorted(PACKAGE.iterdir()):
    if p.is_file(): files[str(p.relative_to(REPO))]=sha(p)
out={'identity':HERE.name,'kernel':'pranavagrawai/containment-q-20260913t211559z','version':1,'files':files}
(HERE/'FINAL-MANIFEST.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'files':len(files),'manifest_sha256':sha(HERE/'FINAL-MANIFEST.json')},indent=2))
