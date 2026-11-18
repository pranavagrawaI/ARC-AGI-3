"""One explicitly authorized current-session required-file pass; never a residual loop."""
import argparse,base64,hashlib,json,os,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE/'collection_tools'))
import primary_bundle,transport,primary,required_files,collector,remote,retry_transport,space

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def verify_freeze():
 frozen=json.loads((HERE/'freeze.json').read_bytes())
 for name,expected in frozen['files'].items():collector.require(digest(HERE/name)==expected,'reviewed source changed '+name)
 for name,expected in frozen['references'].items():collector.require(digest(name)==expected,'reviewed reference changed '+name)
 return digest(HERE/'freeze.json')

def prepare(primary_path,readback):
 raw=Path(primary_path).read_bytes();primary.verify(raw,readback);bundle=primary_bundle.verify(raw)
 source=Path(readback,'source.ipynb').read_bytes();values=transport.literals(transport.cell_source(json.loads(transport.decode(source))))
 plan=required_files.build(bundle,values['PAYLOAD'])
 return raw,bundle,values,plan

def run(authority,primary_path,terminal,destination):
 started=time.monotonic();end=started+7200
 destination=Path(destination);collector.require(not destination.exists(),'fresh destination required')
 frozen=verify_freeze();auth=json.loads(Path(authority).read_bytes())
 collector.require(auth.get('collection_authorized') is True and auth.get('approved_freeze_sha256')==frozen
   and auth.get('primary_sha256')==digest(primary_path) and auth.get('maximum_passes')==1
   and auth.get('caps')==collector.CAPS|dict(GET_slots=40000),'exact separate collection authority required')
 collector.require(space.check(destination.parent)['eligible'],'19GiB simultaneous required-file reserve')
 raw,bundle,values,plan=prepare(primary_path,terminal)
 collector.require(auth.get('manifest_sha256')==hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest(),'reviewed upfront manifest binding')
 expected=remote.validate_readback(terminal);identity={key:expected[key] for key in ('kernel_id','version','source_exact','status','inner_sha256')}
 # One owner/pass across all fresh destinations, even after a partial/error.
 (HERE/'remote-readback').mkdir(exist_ok=True)
 with (HERE/'remote-readback/required-collection-once.json').open('x') as stream:json.dump(dict(authority_sha256=digest(authority),destination=str(destination),identity=identity),stream)
 # Driver retains one shared cooperative end through retries and final persistence.
 collector.require(time.monotonic()<end,'shared driver deadline before API')
 api=retry_transport.RetryRemote(remote.Remote(),[r['name'] for r in plan['rows']],destination,end)
 result=collector.collect(plan['rows'],destination,api,identity=identity,limits=dict(wall_seconds=max(.001,end-time.monotonic())))
 completed=False
 try:
  collector.require(result['complete'] is True,'required-file collection stopped; no automatic residual pass')
  collector.require(result['before']['wire_seconds']==result['after']['wire_seconds'],'terminal quota moved during collection')
  collector.require(api.attempts<=40000 and time.monotonic()<end,'GET/shared wall cap')
  for name,value in (('REQUIRED-MANIFEST.json',plan),('COLLECTION-INPUTS.json',dict(primary_sha256=digest(primary_path),terminal_readback_sha256=digest(Path(terminal)/'READBACK.json'),authority_sha256=digest(authority),freeze_sha256=frozen))):
   with (destination/name).open('x') as stream:json.dump(value,stream,indent=2);stream.flush();os.fsync(stream.fileno())
  completion=dict(schema='PAIR17_REQUIRED_DRIVER_V1',complete=True,started=started,deadline=end,receipt_sha256=digest(destination/'COLLECTION.json'),transport_GETs=api.attempts,returned_at=time.monotonic())
  with (destination/'DRIVER-COMPLETION.json').open('x') as stream:json.dump(completion,stream,indent=2);stream.flush();os.fsync(stream.fileno())
  collector.require(time.monotonic()<end,'driver publication completed late')
  completed=True
 finally:
  if not completed:
   with (destination/'DRIVER-FAILURE.json').open('x') as stream:json.dump(dict(complete=False,no_retry=True,returned_at=time.monotonic()),stream);stream.flush();os.fsync(stream.fileno())
 return result

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('authority');p.add_argument('primary');p.add_argument('terminal_readback');p.add_argument('destination');a=p.parse_args()
 value=run(a.authority,a.primary,a.terminal_readback,a.destination);print(json.dumps(dict(complete=value['complete'],files=len(value['files']))))
