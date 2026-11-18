"""Preparation only; generate candidate bindings before independent review."""
import argparse,json,sys,sysconfig
from pathlib import Path
from common import HERE,hash_file,put
EXCLUDE={'INVENTORY.json','LOCAL-CHECK.json','HERMES-RELEASE.json','PM-CREDENTIAL-APPROVAL.json'}
if (HERE/'live').exists() or (HERE/'independent-review/DECISION.json').exists():raise SystemExit('live/review exists; no regeneration')
files=[p for p in HERE.iterdir() if p.is_file() and p.name not in EXCLUDE]
for directory in ['engine','assets','discovery','donor','preparation']:
 files += [p for p in (HERE/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
site=Path(sysconfig.get_path('purelib'));stdlib=Path(sysconfig.get_path('stdlib'))
runtime=[Path(sys.executable).resolve()]
for module in ['arc_agi','arcengine']:
 runtime += list((site/module).rglob('*.py'))
# Pin the actual standard-library sources staged by the inherited sealed runtime.
sys.path.insert(0,str(HERE/'engine'))
from containment_base import RUNTIME_FILES,RUNTIME_DIRS
runtime += [stdlib/n for n in RUNTIME_FILES]
for d in RUNTIME_DIRS:runtime += list((stdlib/d).rglob('*.py'))
value={'task':'WORLD-MODEL-UTILITY-RETRY','files':{str(p.relative_to(HERE)):hash_file(p) for p in sorted(set(files))},'runtime':{str(p.resolve()):hash_file(p) for p in sorted(set(runtime))},'runtime_scope':'Python executable, official SDK/game engine sources and sealed copied stdlib. Other installed interpreter/native/dependency libraries remain trusted host environment, not checkpoint identity.'}
# Reversible preparation operation, not an approval or paid admission.
(HERE/'INVENTORY.json').write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')
print(hash_file(HERE/'INVENTORY.json'))
