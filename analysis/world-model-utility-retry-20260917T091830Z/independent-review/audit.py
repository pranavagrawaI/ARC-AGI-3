"""Read-only final binding and tested-source preservation audit. No candidate imports."""
import json,hashlib
from pathlib import Path
P=Path(__file__).resolve().parents[1];J=lambda f:json.loads(f.read_bytes());H=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
expected='6222b21ed809bd4e1a41043593e4a594b9390f7a69a18d4dad1dc5213bdeafc7'
assert H(P/'INVENTORY.json')==expected
inv=J(P/'INVENTORY.json')
for n,h in inv['files'].items():assert H(P/n)==h,n
for n,h in inv['runtime'].items():assert H(Path(n))==h,n
for n,h in J(P/'independent-review/early-source-hashes.json').items():assert H(P/n)==h,n
pre=J(P/'preparation/PRE-REPORT-INVENTORY.json')
changed=[n for n,h in pre['files'].items() if inv['files'].get(n)!=h]
assert not [n for n in changed if n.endswith('.py')],changed
assert pre['runtime']==inv['runtime']
d=J(P/'DONOR.json');old=Path(d['donor'])
for a,b in [(P/'donor/FROZEN.json',old/'live/constructor/FROZEN.json'),(P/'donor/prefix.jsonl',old/'live/prefix/child/prefix.jsonl')]:assert a.read_bytes()==b.read_bytes()
f=J(P/'donor/FROZEN.json');assert hashlib.sha256(f['candidate_source'].encode()).hexdigest()==f['source_digest']
assert H(old/'INVENTORY.json')==d['old_inventory_sha256']
for n,h in J(old/'INVENTORY.json')['files'].items():assert H(old/n)==h,n
assert H(old/'live/RESULT.json')==d['old_result_sha256']
l=J(old/'live/ledger.json');assert l['known_cost_usd']==d['old_known_cost_usd'] and l['unknown_exposure_usd']==d['old_unknown_exposure_usd'] and l['stopped']
assert not (P/'live').exists()
print(json.dumps({'status':'PASS','inventory_sha256':expected,'pre_report_inventory_sha256':H(P/'preparation/PRE-REPORT-INVENTORY.json'),'changed_previous_members':changed,'added_members':sorted(set(inv['files'])-set(pre['files'])),'files_verified':len(inv['files']),'runtime_verified':len(inv['runtime']),'tested_code_unchanged':True,'old_ledger_sha256':H(old/'live/ledger.json'),'source_digest':f['source_digest'],'live_absent':True,'bindings':inv},indent=2))
