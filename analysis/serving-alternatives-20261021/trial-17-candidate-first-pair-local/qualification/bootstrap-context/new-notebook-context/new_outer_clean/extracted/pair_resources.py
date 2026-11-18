"""Actual shipping-native log/scope plus six-probe rederivation; no proxy."""
import hashlib,json
from pathlib import Path
from candidate_evidence import require,read,raw,verify_sample

def native_probe(root,sample,binding,scope):
 p=sample['probes']['native'];data=raw(root,p['artifact']);text=data.decode()
 graceful='Gracefully exiting... Remaining number of requests' in text
 crash=any(s in text for s in ('SIGQUIT received.','Triggering SIGQUIT for cleanup...'))
 gone=p['frontend_exited']
 require(type(gone) is bool and (bool(sample['process']['remaining']) or gone),'native exit/process consistency')
 expected=p['log_stable'] is True and gone and graceful and not crash
 require(type(p['log_stable']) is bool and p['graceful_marker']==graceful
   and p['crash_marker']==crash and p['eligible']==expected and p['direct_watchdog_order_observed'] is False,'actual native raw rederivation')
 for record in (p['scope_before'],p['scope_after'],sample['probes']['cgroup']['scope_before'],sample['probes']['cgroup']['scope_after']):
  require(record['input_scope']=='LIVE_BOUND_CGROUP_AND_FRESH_STDERR' and record['run_id']==scope['run_id']
    and record['cgroup']==scope['observer']['cgroup'] and record['namespaces']==scope['observer']['namespaces']
    and record['mount_sha256']==hashlib.sha256(scope['mount_record'].encode()).hexdigest()
    and record['native_marker_sha256']==scope['lease']['marker_sha256'],'fresh native scope/marker binding')
 return expected

def final(root,config,events,capture):
 root=Path(root);release=read(root/'release.json');scope=read(root/'release-scope.json');lease=read(root/'native-lease.json')
 bound=read(root/'control-server-binding.json')['registered_native_binding'];identity=bound['members'][str(bound['pgid'])]
 require((str(identity['pgid']),str(identity['starttime']),identity['boot_id'])==(capture['server_pgid'],capture['server_start'],capture['boot_id']),'final actual control identity')
 starts=[row for row in events if row['event']=='control_cleanup_start'];rs=[row for row in events if row['event']=='release_start'];rr=[row for row in events if row['event']=='release_result']
 require(len(starts)==len(rs)==len(rr)==1,'one configured final native release')
 begin=starts[0];deadline=min(events[0]['work_deadline'],begin['cleanup_start']+180)
 require(begin['phase_deadline']==release['absolute_deadline']==rs[0]['deadline']==deadline
   and capture['end']<=begin['cleanup_start']<=begin['t']<=rs[0]['t']<=release['start']<=release['end']<=rr[0]['t']<=deadline,'configured final180/work deadline')
 require(release['pass'] is True and release['fallback_used'] is False and release['admission_policy']=='FIRST_FRESH_NONHOST'
   and release['error'] is None and release['cleanup_error'] is None and release['controller_query_failure'] is None and release['remaining']==[],'final graceful native release required')
 require(all(rr[0].get(k)==v for k,v in release.items()),'release trace/file equality')
 accepted=[];previous=release['start']
 for index,sample in enumerate(release['samples']):
  require(previous<=sample['start']<=sample['end']<=release['end'],'native sequential samples')
  value=verify_sample(root,sample,bound,capture['gpu_uuid'],deadline,allow_cpu=config['mode']=='CPU_FIXTURE_ONLY',native=True)
  if value['eligible']:accepted.append(index)
  previous=sample['end']
 require(bool(release['samples']) and accepted==[len(release['samples'])-1],'first accepted native final resource sample')
 require(scope['lease']==lease and scope['run_id']==config['run_id'],'same final native lease')
 marker_path=Path(lease['marker_path']).relative_to(Path(config['cwd'])/'output')
 marker=(root/marker_path).read_bytes()
 require(hashlib.sha256(marker).hexdigest()==lease['marker_sha256'] and json.loads(marker)==lease['created'],'independent native marker bytes')
 launch_path=Path(lease['launch_path']).relative_to(Path(config['cwd'])/'output')
 launch_bytes=(root/launch_path).read_bytes();launch=json.loads(launch_bytes)
 require(hashlib.sha256(launch_bytes).hexdigest()==lease['launch_sha256'] and launch['wrapper']==identity
   and launch['marker_sha256']==lease['marker_sha256'] and lease['created']['controller']==bound['controller']
   and lease['created']['created_at']<=launch['start']<=launch['end']<=bound['bound_at'],'independent native launch/controller bytes')
 log_path=Path(lease['created']['log_path']).relative_to(Path(config['cwd'])/'output')
 require((root/log_path).read_bytes()==raw(root,release['samples'][-1]['probes']['native']['artifact']),'final native actual log/snapshot equality')
 # Preserve honest original140 report-only result, including false.
 return dict(final_native_resource=True,original140=release['samples'][-1]['probes']['cgroup']['eligible'],original140_report_only=True,absolute_deadline=deadline)
