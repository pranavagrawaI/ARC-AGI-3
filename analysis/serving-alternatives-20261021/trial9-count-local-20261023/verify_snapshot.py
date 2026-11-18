"""Fast stdlib-only independent audit joins and snapshot verification; no network."""
import hashlib,json,zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path):return json.loads(path.read_text())
def verify():
 prior=BASE/'trial-9-readiness-local';freeze=load(prior/'freeze.json')
 assert sha(prior/'freeze.json')=='336f0c2d6a89951178e247a9a92825a5e6053099bb53694a06a14a2cd6d82d53'
 assert len(freeze['files'])==642
 for name,digest in freeze['files'].items():assert sha(prior/name)==digest,name
 wheel=BASE.parent/'serving-headroom-v3-20261008/evidence/packages/sglang-0.5.19+gd00d88efc8d6-cp312-cp312-linux_x86_64.whl'
 assert sha(wheel)=='d0620216732b82b6efd4f8a78cc9b5f55343a4d3837fdfa3bcbb841b7718a225'
 with zipfile.ZipFile(wheel) as z:
  for name in ('protocol.py','template_manager.py'):
   parent='entrypoints/openai' if name=='protocol.py' else 'parser'
   assert z.read('sglang/srt/'+parent+'/'+name)==(BASE/'trial-5-serving-screen/prompt-audit/source'/name).read_bytes()
 report=load(HERE/'ALL-CORPUS-PROOF.json')
 assert report['catalog_sha256']==sha(HERE/'catalog.exact-wire-v2.json')
 assert report['records_sha256']==sha(HERE/'exact-wire-records.jsonl')
 assert report['source_sha256']['script']==sha(HERE/'all_corpus.py')
 old=load(prior/'catalog.json');new=load(HERE/'catalog.exact-wire-v2.json')
 assert set(old['records'])==set(new['records']) and len(new['records'])==1369
 for key in ('streams','initial_holders','initial_queue','slots'):assert new[key]==old[key]
 for key,record in new['records'].items():
  for field in ('wire_sha','images','cap','game','index','file','offset','extends_previous'):assert record[field]==old['records'][key][field]
 recovery=BASE/'trial9-cutoff-local-20261023/evidence/partial-members'
 inventory=load(recovery/'RECOVERY.json');members={r['name']:r for r in inventory['entries']}
 used=set()
 for observed in report['observed']:
  directory=recovery/'members/serving-wire-capacity9/output/control'/observed['rid']
  for file in directory.iterdir():
   name=str(file.relative_to(recovery/'members'));expected=members[name]
   assert sha(file)==expected['sha256'] and file.stat().st_size==expected['bytes'];used.add(name)
  raw=(directory/'wire.http').read_bytes().split(b'\r\n\r\n',1)[1]
  assert hashlib.sha256(raw).hexdigest()==new['records'][observed['key']]['prepared_body_sha256']==observed['actual_http_body_sha256']
  usage=load(directory/'body.bin')['usage'];assert usage['prompt_tokens']==observed['prompt_tokens']==new['records'][observed['key']]['local_prompt']
  assert usage['prompt_tokens_details']['image_tokens']==observed['image_tokens']==400*len(new['records'][observed['key']]['images'])
 assert len(report['observed'])==275
 assert load(HERE/'render-probe.json')['joined_subset']==275
 assert load(HERE/'diagnostic-probe.json')['scope'].startswith('EXACT_DIAGNOSTIC_WIRE_CPU')
 assert 'Ran 3 tests' in (HERE/'reference-tests.log').read_text() and (HERE/'reference-tests.log').read_text().rstrip().endswith('OK')
 return dict(status='AUDIT_SNAPSHOT_VERIFIED_NO_GPU_AUTHORITY',trial9_bindings_exact=642,records=1369,observed_control_subset=275,missing_completed_joins=266,
             used_recovered_members_sha_size_verified=len(used),pinned_wheel_protocol_loader_bytes_exact=True,
             candidate_runtime_equivalence=False,trial9_stays_invalid=True,ratio=None,successor_freeze_held=True)

if __name__=='__main__':
 result=verify()
 snapshot=HERE/'AUDIT-SNAPSHOT.json'
 if snapshot.exists():
  for name,digest in load(snapshot)['files'].items():assert sha(HERE/name)==digest,name
  result['snapshot_sha256']=sha(snapshot)
 print(json.dumps(result,indent=2))
