import datetime,hashlib,json,os,sys
from pathlib import Path
root=Path.cwd();candidate=root/'analysis/operational-theory-t1r3-20260917'
sys.path.insert(0,str(candidate))
from identity import verify,sha
from run_pair import metadata_check
assert verify()=='992cc66ae2e8577f69f852f00a5330f8f0fb2269155c435f4d1d35ba2931c53c'
assert sha(candidate/'EXPERIMENT-CARD.md')=='20b831a60984974c301eda9eeabe8e1d6dadcdcf8b201c8a7f78861c41f25079'
review=candidate/'T2R3-REVIEW.json'
assert sha(review)=='6c43c41d615db4c4ad1e207f96743bcccda68deab6418b971293403a221b8928'
r=json.loads(review.read_text());assert r['decision']=='PASS' and r['role']=='independent-reviewer' and r['task']=='operational-theory-T1R3' and r['manifest_sha256']==verify()
pre=json.loads((candidate/'PREDECESSOR.json').read_text())
while pre:
 assert all(sha(root/pre['identity']/name)==value for name,value in pre['files'].items())
 pre=pre.get('earlier_chain')
assert not (candidate/'live').exists()
prior=[]
for d in (root/'analysis').glob('operational-theory*'):
 if 'qualification' in d.name:continue
 for f in d.rglob('*.json'):
  if any(part in ('worker-source','actor','canonical') for part in f.parts):continue
  if 'AUTHORIZ' in f.name.upper() or f.name in ('CONSUMED.json','RESULT.json','STOP.json'):
   if '992cc66ae2e8577f69f852f00a5330f8f0fb2269155c435f4d1d35ba2931c53c' in f.read_text():prior.append(str(f))
assert not prior,prior
out=Path('/tmp/arc-t3-operator/output-path').read_text();out=Path(out)
Path('/tmp/arc-t3-operator/output-path').write_text(str(out))
inventory={}
for d in [candidate,root/'analysis/operational-theory-t1-20260917',root/'analysis/operational-theory-t1r1-20260917',root/'analysis/operational-theory-t1r2-20260917']:
 inventory[str(d.relative_to(root))]={str(f.relative_to(d)):sha(f) for f in d.rglob('*') if f.is_file()}
(out/'SOURCE-BEFORE.json').write_text(json.dumps(inventory,sort_keys=True,indent=2)+'\n')
(out/'USER-TASK.txt').write_bytes(Path('/tmp/arc-operational-theory-t3-launch.txt').read_bytes())
metadata_candidates=[]
for f in (candidate/'metadata').glob('*'):
 try: metadata_candidates.append({'path':str(f.relative_to(root)),'sha256':metadata_check(f)})
 except (KeyError,TypeError,ValueError):pass
assert metadata_candidates
record={'manifest_sha256':verify(),'card_sha256':sha(candidate/'EXPERIMENT-CARD.md'),'review_sha256':sha(review),'review_decision':'PASS','predecessor_chain_verified':True,'previous_exact_candidate_launches':prior,'preserved_metadata_gate':metadata_candidates,'credential_present_in_environment':bool(os.environ.get('OPENROUTER_API_KEY')),'network_calls':0}
(out/'PREFLIGHT.json').write_text(json.dumps(record,indent=2)+'\n')
print(str(out));print(json.dumps(record,indent=2))
