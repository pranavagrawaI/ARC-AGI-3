"""Offline evidence integrity check; a verified negative is not a qualification pass."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
def main():
 m=json.loads((HERE/'FINAL-MANIFEST.json').read_text())
 for rel,h in m['files'].items():assert hashlib.sha256((REPO/rel).read_bytes()).hexdigest()==h,rel
 r=json.loads((HERE/'AUDIT.json').read_text())
 print(json.dumps({'evidence_integrity_verified':True,'files_verified':len(m['files']),'final_manifest_sha256':hashlib.sha256((HERE/'FINAL-MANIFEST.json').read_bytes()).hexdigest(),'qualified':r['qualified'],'remote_executed':r['executed'],'remote_unexercised':r['unexercised']},indent=2))
if __name__=='__main__':main()
