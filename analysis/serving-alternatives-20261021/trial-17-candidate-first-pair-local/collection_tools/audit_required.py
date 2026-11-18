"""Offline complete required-file audit; no archive/CD assertion and no remote calls."""
import argparse,base64,hashlib,json,shutil,sys
from pathlib import Path
HERE=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(HERE/'collection_tools'),str(HERE)]
import primary_bundle,transport,primary,collector,remote,publication,run_required,space

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_bytes())

def ledger(directory,plan):
 directory=Path(directory);c=read(directory/'COLLECTION.json');driver=read(directory/'DRIVER-COMPLETION.json')
 collector.require(not (directory/'DRIVER-FAILURE.json').exists() and c['complete'] is True and driver['complete'] is True
   and driver['schema']=='PAIR17_REQUIRED_DRIVER_V1' and driver['deadline']==driver['started']+7200
   and driver['started']<=driver['returned_at']<driver['deadline'] and driver['receipt_sha256']==digest(directory/'COLLECTION.json'),'actual driver completion/receipt')
 allowed={row['name']:row for row in plan['rows']};files={row['name']:row for row in c['files']};waived={row['name'] for row in c['skipped_conditionals']}
 collector.require(len(files)==len(c['files']) and not set(files)&waived and set(files)|waived==set(allowed),'complete required names without expansion')
 collector.require(c['downloaded_bytes']==sum(row['bytes'] for row in files.values())<=8*1024**3 and c['elapsed_seconds']<7200,'required byte/time cap')
 collector.require(c['list_pages']==len(c['pages'])<=300 and all(row['items']<=100 for row in c['pages']),'list cap')
 attempts=[json.loads(row) for row in (directory/'attempts.jsonl').read_bytes().splitlines()]
 collector.require(c['download_attempts']==len(attempts)==len(files) and {row['name'] for row in attempts}==set(files),'single invocation per selected name')
 transport_rows=[json.loads(row) for row in (directory/'transport-attempts.jsonl').read_bytes().splitlines()]
 starts=[row for row in transport_rows if row['event']=='START'];collector.require(len(starts)==driver['transport_GETs']<=40000,'transport GET ledger')
 for name in files:
  selected=[row for row in transport_rows if row['name']==name];begins=[row for row in selected if row['event']=='START']
  collector.require(len(begins) in (1,2) and [r['attempt_in_pass'] for r in begins]==list(range(1,len(begins)+1)),'at most two transport attempts')
  if len(begins)==2:
   errors=[r for r in selected if r['event']=='ERROR']
   collector.require(len(errors)==1 and errors[0]['type']=='ConnectTimeout' and errors[0]['phase']=='GET'
     and errors[0]['response_absent'] is True and errors[0]['received_bytes']==errors[0]['persisted_bytes']==0
     and errors[0]['retry_eligible'] is True and begins[1]['at']>=errors[0]['at']+1,'only zero-byte no-response ConnectTimeout retry')
  ends=[r for r in selected if r['event']=='RESPONSE_CONTEXT_EXITED'];collector.require(len(ends)==1,'successful transport completion')
 for name in waived:
  collector.require(allowed[name]['conditional'] is True and not collector.pending_body_required(read(directory/'members'/Path(name).with_name('http.json')),name),'conditional raw HTTP waiver')
 for label in ('before','after'):
  rb=read(directory/label/'READBACK.json')
  value=remote.validate_readback(directory/label,__import__('datetime').datetime.fromisoformat(rb['typed_quota']['utc']))
  collector.require(value==c[label],'actual source/quota/readback receipt')
 collector.require(c['before']['wire_seconds']==c['after']['wire_seconds'] and c['before']['inner_sha256']==c['after']['inner_sha256'],'before-after quota/source equality')
 return files

def independent_result(path,origin,bundle):
 collector.require(origin['origin']=='INDEPENDENT_CURRENT_SESSION_DOWNLOAD','actual result independent origin required; primary alias forbidden')
 collector.require(not Path(path).is_symlink(),'actual result symlink forbidden')
 actual=read(path);published=json.loads(base64.b64decode(bundle['files']['result.json']['base64'],validate=True))
 collector.require(actual==published,'actual post-publication result differs; bytes preserved, INVALID')
 return actual

def audit(primary_path,collection_dir,destination,collection_exit):
 directory=Path(collection_dir);destination=Path(destination);destination.mkdir(exist_ok=False)
 report=dict(status='INVALID_INCONCLUSIVE',qualification='FEASIBILITY_ONLY',ratio=None,supplemental_complete=False,
   collection_scope='CURRENT_SESSION_UPFRONT_REQUIRED_FILES_ONLY',numeric_archive_or_CD_claim=False,adoption=False,issues=[],dependency_expansion=[])
 try:
  run_required.verify_freeze();collector.require(space.check(destination,stage='assembly')['eligible'],'11GiB fresh assembly/derived reserve')
  raw,bundle,values,plan=run_required.prepare(primary_path,directory/'after')
  collector.require(bundle['missing']==[],'missing primary required entry')
  collector.require(read(directory/'REQUIRED-MANIFEST.json')==plan,'reproducible required manifest')
  inputs=read(directory/'COLLECTION-INPUTS.json');collector.require(inputs['primary_sha256']==digest(primary_path) and inputs['freeze_sha256']==digest(HERE/'freeze.json'),'collection input pins')
  exit_proof=read(collection_exit)
  collector.require(exit_proof['exit_code']==0 and exit_proof['script_sha256']==digest(HERE/'collection_tools/run_required.py') and exit_proof['output_destination']==str(directory.resolve()) and exit_proof['returned']==dict(complete=True,files=len(read(directory/'COLLECTION.json')['files'])),'actual collector CLI return/exit/source agreement')
  driver=read(directory/'DRIVER-COMPLETION.json')
  collector.require(exit_proof['start']<=driver['started']<=driver['returned_at']<=exit_proof['end'] and exit_proof['end']-exit_proof['start']<=7200,'actual collector return/elapsed deadline')
  files=ledger(directory,plan)
  assembly_bytes=sum(row['bytes'] for row in files.values())+sum(row['bytes'] for row in plan['embedded'].values())
  collector.require(assembly_bytes<=8*1024**3,'8GiB fresh evidence assembly cap, including primary embedded bytes')
  evidence=destination/'evidence';evidence.mkdir();origins={}
  for name,row in plan['embedded'].items():
   data=base64.b64decode(bundle['files'][row['primary_key']]['base64'],validate=True);target=evidence/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
   origins[name]=dict(origin='EXACT_PRIMARY_EMBEDDED',sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
  for name,row in files.items():
   source=directory/'members'/name;data=source.read_bytes();collector.require(not source.is_symlink() and row['origin']=='INDEPENDENT_CURRENT_SESSION_DOWNLOAD' and row['complete'] is True and len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256'],'independent required-file hash/origin')
   target=evidence/name;collector.require(not target.exists(),'overlapping primary/fetched origins');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data);origins[name]=dict(row,source=str(source))
  collector.require(space.check(destination,stage='derived')['eligible'],'3GiB derived reserve after assembly')
  cfg=read(HERE/'packet-spec.json')['config'];parent=evidence/Path(cfg['cwd']).name;output=parent/'output'
  actual_name=str(Path(cfg['cwd']).name+'/output/result.json');collector.require(origins[actual_name]['origin']=='INDEPENDENT_CURRENT_SESSION_DOWNLOAD','actual result independent origin required')
  actual=independent_result(output/'result.json',origins[actual_name],bundle)
  for name,row in values['PAYLOAD'].items():collector.require((parent/name).read_bytes()==base64.b64decode(row['base64'],validate=True),'actual payload source '+name)
  collector.require(read(parent/'config.json')==cfg and read(parent/'assignment.json')==values['ASSIGNMENT'],'config/assignment source')
  events=[json.loads(row) for row in (output/'supervisor.jsonl').read_bytes().splitlines()]
  published=primary_bundle.verify_publication(output,raw,events[0]['work_deadline'],cfg['primary_seconds'])
  import reduce_api
  pair=reduce_api.compare(output)
  collector.require(not pair['issues'] and pair['ratio'] is not None and actual['pass_'] is True
    and actual['qualification']==pair['verdict'] and actual['ratio']==pair['ratio'],'all independent raw/scientific pair gates')
  collector.require(not any(actual.get(key) for key in ('cleanup_errors','phase_cleanup_errors','retained_queries','surviving_phase_groups','surviving_native_group','primary_bundle_error','late_work')),'actual lifecycle/publication failure')
  for row in bundle['health_inventory']:
   data=(output/row['path']).read_bytes();collector.require(len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256'],'actual bounded health raw hash')
  proof=dict(status='PAIRED_CAPACITY_QUALIFIED',qualification=pair['verdict'],ratio=pair['ratio'],pair=pair,publication=published,required_files=len(files),independent_actual_result=True)
  return publication.publish(destination,report,proof,origins)
 except FileNotFoundError as error:
  missing=Path(error.filename);base=destination/'evidence'
  report['dependency_expansion'].append(dict(path=error.filename,remote_name=str(missing.relative_to(base)) if missing.is_relative_to(base) else None,reason='required raw checker dependency missing; no automatic fetch expansion'))
  return publication.failed(destination,report,error)
 except Exception as error:return publication.failed(destination,report,error)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('primary');p.add_argument('collection');p.add_argument('destination');p.add_argument('collection_exit');a=p.parse_args()
 value=audit(a.primary,a.collection,a.destination,a.collection_exit);print(json.dumps(value,sort_keys=True));completed=publication.read_completed(a.destination)
 sys.exit(0 if value==completed and value.get('supplemental_complete') and not value.get('issues') and not value.get('dependency_expansion') and value.get('status')=='PAIRED_CAPACITY_QUALIFIED' else 1)
