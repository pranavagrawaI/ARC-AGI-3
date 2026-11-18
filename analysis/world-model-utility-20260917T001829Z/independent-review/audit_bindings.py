"""Read-only package and donor/evidence binding audit; no candidate imports."""
import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parents[1];repo=P.parents[1]
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def j(p):return json.loads(p.read_bytes())
inv=j(P/'INVENTORY.json')
for n,v in inv['files'].items():assert not (P/n).is_symlink() and h(P/n)==v,n
for n,v in inv['runtime'].items():assert h(Path(n))==v,n
donor=j(P/'DONOR-HASHES.json')
for n,v in donor.items():assert h(repo/n)==v,n
evidence=j(P/'EVIDENCE-HASHES.json')
for n,v in evidence.items():assert h(P/n)==v,n
assert not (P/'live').exists()
print(json.dumps({'inventory_sha256':h(P/'INVENTORY.json'),'card_sha256':h(P/'EXPERIMENT-CARD.md'),'files_verified':len(inv['files']),'runtime_verified':len(inv['runtime']),'donor_files_verified':len(donor),'evidence_files_verified':len(evidence),'live_absent':True,'bindings':inv},indent=2))
