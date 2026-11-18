import hashlib,importlib.util,json,re,sys
from pathlib import Path,PurePosixPath
OUT=Path(__file__).resolve().parent;BASE=OUT.parent;ROOT=BASE.parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
review=BASE/'version-provenance-review';mapping=json.loads((review/'session-mapping.json').read_text());source=review/'exact-observation-1789558948333208545/model-runtime/official-launcher-result.json';assert sha(source)==mapping['source_sha256']
receipt=BASE/'r10a-cd82-20260916/push-56e324f3e2004081979fd2f2a9c40e4f.json';assert sha(receipt)=='40ce606c0f820cb8523a739df46b2ddbe5333b8e64126808c33d3e50a8a1ec9c'
assert 'Kernel version 1 successfully pushed.  Please check progress at https://www.kaggle.com/code/pranavagrawai/r10a-cd82-constructor-repair' in json.loads(receipt.read_text())['stdout']
sp=importlib.util.spec_from_file_location('collector_plan',BASE/'r10a-verification/collect_plan.py');mod=importlib.util.module_from_spec(sp);sp.loader.exec_module(mod);d=json.loads(source.read_text());planned=mod.paths(d);assert planned==mapping['planned_paths'];assert len(planned)==60
assert [(g['arm'],g['workspace'],g['game_id']) for g in d['games']]==[('A','arm-A','cd82-fb555c5d'),('B','arm-B','cd82-fb555c5d')]
allowed={}
def add(name,why,limit=8*1024*1024):
 p=PurePosixPath(name);assert not p.is_absolute() and '..' not in p.parts and str(p)==name and '\\' not in name
 assert not any(s in p.parts for s in ['.env','.ssh','.aws','.kaggle','credentials','secrets','server-environment.json'])
 allowed[name]={'path':name,'basis':why,'max_bytes':limit}
for name in planned:
 assert re.fullmatch(r'model-runtime/solver-workspaces/arm-[AB]/(?:cd82-fb555c5d-[a-z0-9]+|recordings)/.+',name)
 add(name,'exact numeric-version launcher session enumeration',32*1024*1024)
for g in d['games']:
 for s in g['sessions']:
  assert s['session_dir'] in [x for x in next(m for m in mapping['sessions'] if m['arm']==g['arm'])['session_dirs']]
  prefix=f"model-runtime/solver-workspaces/{g['workspace']}/{s['session_dir']}"
  for suffix in ['FROZEN.json','RESULT.json','CONSTRUCTION.json','constructor-job/FROZEN.json','constructor-job/RESULT.json']:
   add(prefix+'/'+suffix,'explicit bounded authoritative-root/forensic-child freeze probe; 404 preserved',2*1024*1024)
for name in ['GPU.json','SEAL.json','GAME-PREFLIGHT.json','SHUTDOWN-REAP.json','PROCESS-GUARD.json','BOOTSTRAP-FAILURE.json']:
 add('actor-critic-construction/'+name,'donor_bootstrap/gameplay/process_guard producer receipt',2*1024*1024)
for name in ['official-launcher-result.json','result.json','model-verification.json','backend-facts.json','shutdown.json','system-before.json','system-after.json','request-count-facts.json','launcher-budget.json','warmup-facts.json','readiness.json','model-api-readiness.json','lifecycle-target-launch.json','lifecycle-signals.json','official-launcher-command.json','server-command.json']:
 add('model-runtime/'+name,'official_launcher_runtime fixed receipt',8*1024*1024)
for name in ['MODEL-IDENTITY.json','GAME-PREFLIGHT.json','SEAL.json','GPU.json','ARM-DIFF.json','RESULTS.json','FRAMEWORK.json','requests.jsonl','logical-requests.jsonl','responses.jsonl','game-progress.jsonl','environment-steps.jsonl','budget-allocations.jsonl','cleanup.jsonl','errors.jsonl','delivery.jsonl','local-rejections.jsonl','admission-stops.jsonl','latency.jsonl']:
 add('model-runtime/solver-workspaces/raw/'+name,'gameplay_play/local_accounting known producer receipt',32*1024*1024)
n=d['local_requests']['admitted'];assert n==105;assert [r['call'] for r in d['local_requests']['receipts']]==list(range(1,n+1))
for i in range(1,n+1):
 add(f'model-runtime/solver-workspaces/raw/raw-requests/{i:03d}.wire.json','launcher enumerated call id + gameplay_play request filename formula',4*1024*1024)
 add(f'model-runtime/solver-workspaces/raw/raw-http/{i:02d}.body','launcher enumerated call id + local_accounting response filename formula',2*1024*1024)
assert len(allowed)<=340
protected={str(p.relative_to(ROOT)):sha(p) for p in review.rglob('*') if p.is_file()}
manifest=BASE/'r10a-cd82-20260916/manifest.json';m=json.loads(manifest.read_text())
for name,h in m['sources'].items():assert sha(ROOT/name)==h;protected[name]=h
for name in ['manifest.json','authorization.json','launch-review.json','hermes-release.json','CONSUMED.json']:
 p=BASE/'r10a-cd82-20260916'/name;protected[str(p.relative_to(ROOT))]=sha(p)
for name,data in [('PROTECTED-INPUT-HASHES.json',protected),('ALLOWLIST.json',{'target':'pranavagrawai/r10a-cd82-constructor-repair','version_number':1,'launcher_sha256':sha(source),'push_receipt_sha256':sha(receipt),'max_requests':340,'max_total_bytes':512*1024*1024,'files':list(allowed.values())})]:
 with (OUT/name).open('x') as f:json.dump(data,f,indent=2)
print('Validated',len(allowed),'exact paths; 60 launcher-derived paths, 105 request/response pairs. No network.')
