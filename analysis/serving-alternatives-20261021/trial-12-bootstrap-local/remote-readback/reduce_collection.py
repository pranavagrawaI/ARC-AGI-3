"""Read-only CPU reduction of immutable release12 collection; separate derived output."""
import argparse,base64,csv,hashlib,io,json,sys,time,zipfile,zlib
from pathlib import Path
P=Path(__file__).resolve().parent;HERE=P.parent;sys.path.insert(0,str(HERE))
import transport,artifacts,resources
sha=lambda b:hashlib.sha256(b).hexdigest()
def refs(value):
 if isinstance(value,dict):
  if {'path','sha256','bytes','complete'}<=value.keys():yield value
  else:
   for v in value.values():yield from refs(v)
 elif isinstance(value,list):
  for v in value:yield from refs(v)
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);a=parser.parse_args();out=a.output.resolve();out.mkdir(exist_ok=False)
 started=time.monotonic();freeze=json.loads((HERE/'freeze.json').read_text())
 for n,h in freeze['files'].items():assert sha((HERE/n).read_bytes())==h,n
 for n,h in freeze['references'].items():assert sha(Path(n).read_bytes())==h,n
 terminal=json.loads((P/'terminal-source-status-quota.json').read_text());assert terminal['status']['status']=='ERROR' and terminal['metadata_version_identity_verified']
 assert terminal['accepted_kernel_id']==140429120 and terminal['accepted_version']==1
 remote=(P/'source-terminal.ipynb').read_bytes();local=(HERE/'kernel/serving-release12-20261023.ipynb').read_bytes();assert transport.cell_source(json.loads(remote))==transport.cell_source(json.loads(local))
 inner=transport.decode(remote);marker=json.loads((HERE/'launch-attempt.json').read_text());assert sha(inner)==marker['inner_notebook_sha']
 literals=transport.literals(transport.cell_source(json.loads(inner)))
 archive=P/'output-v1-kernel140429120.zip';collected=json.loads((P/'archive-collection.json').read_text());assert sha(archive.read_bytes())==collected['archive_sha256']
 inventory={};extraction=P/'archive-extracted'
 with zipfile.ZipFile(archive) as zipped:
  assert zipped.testzip() is None;assert len(zipped.namelist())==len(set(zipped.namelist()))
  for info in zipped.infolist():
   data=zipped.read(info);assert len(data)==info.file_size and zlib.crc32(data)&0xffffffff==info.CRC
   assert (extraction/info.filename).read_bytes()==data
   inventory[info.filename]=dict(bytes=len(data),sha256=sha(data),CRC=info.CRC)
 root=extraction/'serving-release12';output=root/'output';raw=output/'query-artifacts'
 for n,record in literals['PAYLOAD'].items():assert (root/n).read_bytes()==base64.b64decode(record['base64']),n
 assert json.loads((root/'config.json').read_text())==literals['CONFIG'];assert json.loads((root/'assignment.json').read_text())==literals['ASSIGNMENT']
 assert (root/'.one-attempt').read_text()==literals['ASSIGNMENT']['serialized_assignment']
 result=json.loads((output/'result.json').read_text());release=json.loads((output/'release.json').read_text());pre=json.loads((output/'pre-release.json').read_text());compat=json.loads((output/'query-compatibility.json').read_text())
 checked=0;unique=set()
 for value in (release,pre,compat):
  for ref in refs(value):
   assert not Path(ref['path']).is_absolute() and '..' not in Path(ref['path']).parts
   artifacts.verify(raw,ref,time.monotonic()+5);checked+=1;unique.add((ref['path'],ref['sha256']))
 trace=[json.loads(line) for line in (output/'supervisor.jsonl').read_text().splitlines()]
 hardware=next(x for x in trace if x['event']=='hardware_pass');uuid=hardware['uuid']
 assert hardware['name']==resources.GPU_NAME and hardware['memory_mib']>=90000
 health={phase:json.loads((output/f'control-package/control-health-{phase}.json').read_text()) for phase in ('identity','serve')}
 for value in health.values():assert value['raw']=={'SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION':'1','SGLANG_HEALTH_CHECK_TIMEOUT':'20'} and value['effective']=={'generation_enabled':True,'timeout_s':20}
 assert health['identity']['native_environ_sha256']==health['serve']['native_environ_sha256']
 rows=[];queries=[]
 def rawbytes(ref):return (raw/ref['path']).read_bytes()
 for name,command in compat.items():queries.append(dict(phase='compatibility',name=name,**command))
 for i,sample in enumerate([pre]+release['samples']):
  phase='pre' if i==0 else 'release';index=i-1
  for name,command in sample['controller_queries']['commands'].items():
   assert command['ok'] and command['cleanup_verified'] and command['child_reaped'] and command['containment']['verified']
   assert command['start']<=command['end']<=command['absolute_deadline'];queries.append(dict(phase=phase,index=index,name=name,**command))
  ps=sample['probes'];gpu=resources.gpu(rawbytes(ps['gpu']['command']['artifacts']['stdout']).decode(),uuid)
  assert gpu['used_mib']==ps['gpu']['used_mib'] and gpu['eligible']==ps['gpu']['eligible']
  assert (not rawbytes(ps['apps']['command']['artifacts']['stdout']).strip())==ps['apps']['eligible']
  hostraw={k:rawbytes(v).decode().strip() for k,v in ps['cgroup']['artifacts'].items()}
  mem={k:int(v.split()[0])*1024 for k,v in (line.split(':',1) for line in hostraw['meminfo'].splitlines())}
  stat={k:int(v) for k,v in (line.split() for line in hostraw['stat'].splitlines())}
  computed=mem['MemAvailable']
  if hostraw['maximum'].isdigit():computed=min(computed,max(0,int(hostraw['maximum'])-int(hostraw['current'])+stat.get('inactive_file',0)))
  assert computed==ps['cgroup']['computed_bytes'] and ps['cgroup']['threshold_bytes']==140*1024**3
  assert (computed>=140*1024**3 and ps['cgroup']['files_stable_during_read'])==ps['cgroup']['eligible']
  disk=json.loads(rawbytes(ps['disk']['artifact']));assert (disk['free_bytes']>=100*1024**3)==ps['disk']['eligible']
  port=json.loads(rawbytes(ps['port']['artifact']));assert port['eligible']==ps['port']['eligible'] and port['address']==['127.0.0.1',8001]
  native=rawbytes(ps['native']['artifact']).decode();assert ('Gracefully exiting... Remaining number of requests' in native)==ps['native']['graceful_marker']
  assert any(s in native for s in ('SIGQUIT received.','Triggering SIGQUIT for cleanup...'))==ps['native']['crash_marker']
  nonhost=sample['process']['eligible'] and all(ps[k]['eligible'] for k in ('gpu','apps','disk','port','native'))
  if phase=='release':
   assert sample['end']<=release['absolute_deadline']
   rows.append(dict(index=index,start=sample['start'],end=sample['end'],process=sample['process'],gpu_mib=gpu['used_mib'],host_bytes=computed,
    host_input=dict(maximum=hostraw['maximum'],current=int(hostraw['current']),MemAvailable=mem['MemAvailable'],file=stat.get('file'),anon=stat.get('anon'),inactive_file=stat.get('inactive_file'),active_file=stat.get('active_file')),
    eligible={k:ps[k]['eligible'] for k in ps},nonhost_join=nonhost,qualifies=sample['eligible']))
 assert len(rows)==85 and not any(r['qualifies'] for r in rows)
 assert release['pass'] is False and release['error'] is None and release['cleanup_error'] is None and release['remaining']==[]
 assert result['qualification']=='NOT_QUALIFIED' and result['surviving_native_group']==[] and result['surviving_phase_groups']==[] and result['retained_queries']==[]
 joined=[r for r in rows if r['nonhost_join']];assert len(joined)==82 and all(r['gpu_mib']==0 and not r['eligible']['cgroup'] for r in joined)
 first=joined[0];term=release['events'][0]['start'];assert len(release['events'])==1 and release['events'][0]['signal']==15
 ready=next(x for x in trace if x['event']=='ready')
 summary=dict(verdict='NOT_QUALIFIED_ORIGINAL_HOST_GATE_UNMET',result=result,release_error=release['error'],cleanup_error=release['cleanup_error'],controller_query_failure=release['controller_query_failure'],
  samples=85,eligible_counts={k:sum(r['eligible'][k] for r in rows) for k in rows[0]['eligible']},joined_nonhost_timely_samples=len(joined),
  host_min_bytes=min(r['host_bytes'] for r in rows),host_max_bytes=max(r['host_bytes'] for r in rows),host_threshold_bytes=140*1024**3,
  first_nonhost_join=dict(index=first['index'],start=first['start'],end=first['end'],start_after_TERM_s=first['start']-term,end_after_TERM_s=first['end']-term),
  signals=release['events'],fallback_flag=release['fallback_used'],KILL_events=0,release_start=release['start'],absolute_release_end=release['absolute_deadline'],release_finish=release['end'],finish_overrun_s=release['end']-release['absolute_deadline'],
  actual_ready=ready,actual_hardware=hardware,actual_health=health,
  queries_total=len(queries),query_timings={name:dict(count=len(v:=[q for q in queries if q['phase']=='release' and q['name']==name]),min_seconds=min(q['end']-q['start'] for q in v),max_seconds=max(q['end']-q['start'] for q in v)) for name in ('gpu','apps')},
  limitations=['Post-load only; no replay/candidate/paired capacity qualification','Initial three shutdown samples have expected transient ineligible components; host remains ineligible in all85','Observation does not identify why host headroom is low or establish active_file reclaimability','No direct watchdog-stop/GPU-context-destruction proof from native text','No threshold/cache changes or causal driver-lag/leak claim'])
 for name,value in (('ARCHIVE-INVENTORY.json',inventory),('SAMPLE-SERIES.json',rows),('QUERY-SERIES.json',queries),('RAW-REDUCTION.json',summary)):
  (out/name).write_text(json.dumps(value,indent=2)+'\n')
 proof=dict(kernel_ref='pranavagrawai/serving-release12-20261023',kernel_id=140429120,version=1,
  freeze_sha256=sha((HERE/'freeze.json').read_bytes()),frozen_inputs=len(freeze['files']),frozen_refs=len(freeze['references']),exact_payloads=len(literals['PAYLOAD']),config_assignment_exact=True,remote_cell_and_inner_exact=True,
  archive_sha256=collected['archive_sha256'],archive_bytes=archive.stat().st_size,members=len(inventory),CRC_and_extracted_bytes_exact=True,
  raw_refs_checked=checked,unique_raw_refs=len(unique),primary_refs={str(p.relative_to(extraction)):sha(p.read_bytes()) for p in output.rglob('*') if p.is_file()},
  typed_terminal_quota=terminal['quota'],baseline_used_s=67703.254,delta_used_s=round(terminal['quota']['used_s']-67703.254,3),
  verdict=summary['verdict'],reducer_sha256=sha(Path(__file__).read_bytes()),seconds=time.monotonic()-started,
  derived_inventory={p.name:sha(p.read_bytes()) for p in out.iterdir() if p.is_file()})
 (out/'COLLECTION-PROOF.json').write_text(json.dumps(proof,indent=2)+'\n')
 print(json.dumps({k:v for k,v in proof.items() if k not in ('primary_refs','derived_inventory')},indent=2))
if __name__=='__main__':main()
