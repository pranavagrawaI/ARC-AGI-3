"""Preparation inventory, not an independent approval."""
from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parent
files=[f for f in P.iterdir() if f.is_file() and f.name not in {'INVENTORY.json','HANDOFF.md','LOCAL-CHECK.json'} and not f.name.startswith('tests-')]
files+=list((P/'engine').glob('*.py'))+list((P/'recorded').rglob('*.json'))+list((P/'recorded').glob('*.jsonl'))
value={'schema':'constructor-prediction-test-inventory-v1','task':'CONSTRUCTOR-PREDICTION-TEST','paid_execution_authorized':False,'files':{str(f.relative_to(P)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(set(files))}}
raw=(json.dumps(value,indent=2,sort_keys=True)+'\n').encode();(P/'INVENTORY.json').write_bytes(raw)
print(hashlib.sha256(raw).hexdigest(),len(value['files']))
