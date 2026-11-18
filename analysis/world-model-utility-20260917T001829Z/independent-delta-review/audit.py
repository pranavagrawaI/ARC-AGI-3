"""Read-only delta, archive, runtime, old review and evidence verification."""
import json,hashlib,difflib
from pathlib import Path
P=Path(__file__).resolve().parents[1];R=P.parents[1]
J=lambda p:json.loads(p.read_bytes())
H=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
oldsha='1306bf4aa80c0e4a5cf9ae7bbc089c66ede8427c3634e004a68037001c35599d';newsha='ac86862b5de9ddb3598205580417655ce3349ba3bc9309ee793e9b15892ea71b'
A=P/'superseded'/oldsha
assert H(A/'INVENTORY.json')==oldsha and H(P/'INVENTORY.json')==newsha
old=J(A/'INVENTORY.json');new=J(P/'INVENTORY.json');changed=[]
for n,h in old['files'].items():
 assert H(A/'candidate'/n)==h,n
 if H(P/n)!=h:changed.append(n)
for n,h in new['files'].items():assert H(P/n)==h,n
for n,h in new['runtime'].items():assert H(Path(n))==h,n
assert old['runtime']==new['runtime']
assert H(P/'EXPERIMENT-CARD.md')==old['files']['EXPERIMENT-CARD.md']
for n in ['actor.py','model.py','game.py','worker.py','reconcile.py','common.py','auth.py','protocol.py','request_contract.py']:
 assert H(P/n)==old['files'][n],n
for n,h in J(P/'DONOR-HASHES.json').items():assert H(R/n)==h,n
review=J(A/'REVIEW-HASHES.json')
for n,h in review.items():assert H(P/n)==h,n
sup=J(A/'SUPERSEDED.json')
assert H(P/'independent-review/DECISION.json')==sup['review_decision_sha256']
assert H(P/'independent-review/REPORT.md')==sup['review_report_sha256']
for n,h in J(P/'EVIDENCE-HASHES.json').items():assert H(P/n)==h,n
assert not (P/'live').exists()
expected=['run_utility.py','transport.py','test_utility.py','make_inventory.py']
assert sorted(n for n in changed if n.endswith('.py'))==sorted(expected)
delta=''.join(''.join(difflib.unified_diff((A/'candidate'/n).read_text().splitlines(True),(P/n).read_text().splitlines(True),fromfile='superseded/'+n,tofile='candidate/'+n)) for n in expected)
assert delta==(P/'REPAIR-DELTA.diff').read_text()
print(json.dumps({'status':'PASS','old_inventory_sha256':oldsha,'inventory_sha256':newsha,'card_sha256':H(P/'EXPERIMENT-CARD.md'),'changed_old_members':changed,'added_members':sorted(set(new['files'])-set(old['files'])),'inventoried_files_verified':len(new['files']),'archived_members_verified':len(old['files']),'runtime_files_verified':len(new['runtime']),'original_review_files_verified':len(review),'evidence_files_verified':len(J(P/'EVIDENCE-HASHES.json')),'exact_executable_diff_verified':True,'live_absent':True,'bindings':new},indent=2))
