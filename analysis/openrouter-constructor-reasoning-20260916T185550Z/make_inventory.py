"""Preparation inventory only, not a review/approval or production freeze."""
from pathlib import Path
import hashlib,json
P=Path(__file__).resolve().parent
if (P/'live').exists():raise SystemExit('consumed: never rebuild')
files=[p for p in P.iterdir() if p.is_file() and p.name not in {'INVENTORY.json','HANDOFF.md','LOCAL-CHECK.json'} and not p.name.startswith('tests-')]
files += list((P/'engine').glob('*.py'))+list((P/'discovery').glob('*'))
files=sorted(set(files))
value={'schema':'or-pair-candidate-inventory-v1','task':'OR-CONSTRUCTOR-REASONING-PAIR','endpoint':'reka/fp8','files':{str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
raw=(json.dumps(value,indent=2,sort_keys=True)+'\n').encode()
(P/'INVENTORY.json').write_bytes(raw)
print(hashlib.sha256(raw).hexdigest(),len(files))
