"""Verify BOTH immutable audit dictionaries plus actual local import pins."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def paths(root=HERE):
 root=Path(root);value=json.loads((root/'audit-bindings.json').read_text());result={root/'audit-bindings.json'}
 for name,expected in value['files'].items():
  path=(root/name).resolve()
  if sha(path)!=expected:raise ValueError('audit/dependency binding changed '+name)
  result.add(path)
 for name,expected in value['snapshots'].items():
  path=(root/name).resolve()
  if sha(path)!=expected:raise ValueError('audit snapshot identity changed '+name)
  result.add(path);snapshot=json.loads(path.read_text())
  for section in ('files','references'):
   for file,digest in snapshot[section].items():
    ref=(path.parent/file).resolve()
    if sha(ref)!=digest:raise ValueError('audit snapshot '+section+' drift '+file)
    result.add(ref)
 inventory=json.loads((root/'evidence/actual-audit-imports.json').read_text())
 for record in inventory['imports'].values():
  path=Path(record['path']).resolve()
  if path not in result or sha(path)!=record['sha256']:raise ValueError('actual imported audit dependency drift '+str(path))
 return result

def verify(root=HERE):
 result=paths(root);return {'status':'ALL_AUDIT_FILES_REFERENCES_ACTUAL_IMPORTS_EXACT','bindings':len(result)}
if __name__=='__main__':print(json.dumps(verify(),indent=2))
