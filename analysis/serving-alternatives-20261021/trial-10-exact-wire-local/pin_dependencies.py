"""Capture actual LOCAL audit imports using the cached CPU environment only."""
import hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;REPO=BASE.parents[1]
sys.path.insert(0,str(BASE/'trial-5-serving-screen/prompt-audit'))
import audit,candidate_tools
imports={}
for name,module in sorted(sys.modules.items()):
 path=getattr(module,'__file__',None)
 if not path:continue
 path=Path(path).resolve()
 if path.is_relative_to(REPO) and path.suffix=='.py':
  imports[name]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
assert Path(imports['corpus']['path'])==BASE/'trial-4-saturated-local/corpus.py'
assert hashlib.sha256((HERE/'corpus_base.py').read_bytes()).hexdigest()==imports['corpus']['sha256']
value={'scope':'ACTUAL_LOCAL_CPU_AUDIT_IMPORTS; standardlib and external distributions identified by version in accepted audit', 'imports':imports}
(HERE/'evidence/actual-audit-imports.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value,indent=2))
