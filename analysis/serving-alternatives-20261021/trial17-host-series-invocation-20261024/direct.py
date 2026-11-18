"""One approved direct frozen-function invocation; no transport wrapper or retries."""
import datetime as dt,hashlib,json,os,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent
ROOT=BASE/'trial-17-candidate-first-pair-r2-local'
sys.path[:0]=[str(ROOT/'collection_tools'),str(ROOT)]
import collector,remote,space
def sha(path):
 with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
started=time.monotonic()
authority=BASE/'coordination-20261024/trial17-one-host-series-remote-approved.json'
manifest_path=BASE/'coordination-20261024/trial17-one-host-series-manifest.json'
assert sha(authority)=='52911eee6d99e19009140b824f44718f7c8795dac7f8f016ccb7088975476b92'
assert sha(manifest_path)=='37717363767bb66d7010d06e56277710465bfcb82e7e73999167c658b2d812a6'
auth=json.loads(authority.read_bytes());manifest=json.loads(manifest_path.read_bytes())
assert auth['collection_authorized'] is True and auth['maximum_invocations']==1 and auth['retry_authorized'] is False
assert auth['limits']==dict(list_pages=100,page_size=100,downloads=1,bytes=28*1024**2,wall_seconds=600,concurrency=1)
assert len(manifest)==1 and manifest[0]['name']=='serving-candidate-first-capacity17/output/host-series.jsonl'
assert manifest[0]['expected_bytes']==27510319 and manifest[0]['expected_sha256']=='9b04bd666d43476cdb4d8cc3bf23c769da93d5d68db29e86b54fb55416246432'
for path,digest in auth['source_pins'].items():assert sha(path)==digest,path
assert sha(ROOT/'freeze.json')==auth['approved_freeze_sha256']
frozen=json.loads((ROOT/'freeze.json').read_bytes())
for name,digest in frozen['files'].items():assert sha(ROOT/name)==digest,name
for name,digest in frozen['references'].items():assert sha(name)==digest,name
dest=Path(auth['destination']);assert not dest.exists(),'exclusive destination already exists'
active=[]
for path in Path('/proc').iterdir():
 if not path.name.isdigit() or int(path.name)==os.getpid():continue
 try:
  args=[x.decode() for x in (path/'cmdline').read_bytes().split(b'\0') if x]
  if str(Path(__file__).resolve()) in args or str(dest) in args:active.append(path.name)
 except OSError:pass
assert not active,active
preflight=space.check(ROOT);assert preflight['eligible'],preflight
with (HERE/'PREFLIGHT.json').open('x') as stream:json.dump(dict(space=preflight,authority_sha256=sha(authority),manifest_sha256=sha(manifest_path),source_pins=auth['source_pins'],frozen_bindings_verified=len(frozen['files'])+len(frozen['references']),active_same_file_collectors=active),stream,indent=2)
with (HERE/'INVOCATION.json').open('x') as stream:json.dump(dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),pid=os.getpid(),command=sys.argv,interpreter=sys.executable,outer_cap_seconds=660,collector_cap_seconds=600,retry_wrapper=False,maximum_invocations=1),stream,indent=2)
value=collector.collect(manifest,dest,remote.Remote(),limits=auth['limits'],identity=auth['identity'])
with (HERE/'FUNCTION-RETURNED.json').open('x') as stream:json.dump(value,stream,indent=2)
print(json.dumps(value,indent=2),flush=True)
if not value['complete']:raise SystemExit(1)
assert value['download_attempts']==1 and value['downloaded_bytes']==27510319 and len(value['files'])==1
assert value['before']['wire_seconds']==value['after']['wire_seconds']==dict(used_s='4842.678',reserved_s='0',total_allowed_s='108000',minimum_allowed_s='108000'),'terminal Decimal quota changed'
assert json.loads((dest/'COLLECTION.json').read_bytes())==value,'actual returned/saved receipt disagreement'
with (HERE/'DIRECT-COMPLETION.json').open('x') as stream:json.dump(dict(complete=True,returned_at=time.monotonic(),outer_started=started,outer_elapsed_seconds=time.monotonic()-started,collection_sha256=sha(dest/'COLLECTION.json'),returned_sha256=sha(HERE/'FUNCTION-RETURNED.json'),frozen_bindings_verified=len(frozen['files'])+len(frozen['references']),one_GET=True,retries=0),stream,indent=2)
