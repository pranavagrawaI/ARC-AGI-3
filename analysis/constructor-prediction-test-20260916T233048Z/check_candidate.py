"""Read-only integrity checks. No credentials, networking, scoring or model calls."""
import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
def main():
 inv=json.loads((P/'INVENTORY.json').read_text())
 for n,h in inv['files'].items():
  if n=='recorded/heldout/outcome.json':continue  # Target bytes are verified by bound scoring.
  assert sha(P/n)==h,n
 for n,h in json.loads((P/'PRIOR-HASHES.json').read_text()).items():assert sha(Path(n))==h,n
 selected=json.loads((P/'recorded/SELECTION.json').read_text())
 assert sha(P/'EXPERIMENT-CARD.md')==selected['card_sha256']
 assert sha(P/'recorded/prefix.jsonl')==selected['prefix_sha256']
 assert sha(P/'recorded/query.json')==selected['query_sha256']
 # Do not open heldout contents in this general verifier; bound scoring owns that read.
 assert not (P/'auth.py').exists() and not (P/'run_pair.py').exists()
 print(json.dumps({'status':'VERIFIED_LOCAL_CANDIDATE','inventory_sha256':sha(P/'INVENTORY.json'),'selection_card_unchanged':True,'target_file_verification':'deferred to bound scorer','prior_files_unchanged':len(json.loads((P/'PRIOR-HASHES.json').read_text())),'model_calls':0,'credentials_accessed':False,'scope':'integrity verification, not independent approval'},indent=2))
if __name__=='__main__':main()
