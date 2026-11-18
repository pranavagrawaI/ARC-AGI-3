"""Reverse producer/consumer joins using accepted within-arm and raw resource checks."""
import hashlib,json
from pathlib import Path
import candidate_evidence as candidate
import candidate_resources,handoff,pair_resources,artifacts,time
require=candidate.require
read=candidate.read
LABELS=['before_candidate_preparation','corpus.prepare.before','corpus.prepare.after',
 'candidate.prepare.before','candidate.prepare.after','candidate.identity.before','candidate.identity.after',
 'candidate.server.spawned','candidate.health200','candidate.replay.before','candidate.replay.after','candidate.qualify.after',
 'candidate.cleaned','candidate.released.before_fresh','control.prepare.after','control.identity.before','control.identity.after','control.health200',
 'control.replay.before','control.replay.after','control.qualify.before','control.qualify.after','control.released','final_cleanup']

def replay_join(root,events,label,capture):
 starts=[r for r in events if r['event']=='spawn' and r.get('phase')==label+'.replay'];exits=[r for r in events if r['event']=='exit' and r.get('phase')==label+'.replay']
 bound=[r for r in events if r['event']=='capacity_server_bound' and r['arm']==label]
 require(len(starts)==len(exits)==len(bound)==1,'one actual replay per arm')
 spawn=starts[0];exit=exits[0];identity=spawn['identity']
 header=json.loads((root/label/'events.jsonl').read_bytes().splitlines()[0])
 require(identity is not None and header['pid']==header['replay_pgid']==spawn['pid']==spawn['pgid']==exit['pid']==identity['pid']==identity['pgid']
   and identity['starttime']==header['replay_starttime'] and identity['boot_id']==header['boot_id']==capture['boot_id'],'actual replay PID/starttime/PGID/boot join')
 server=bound[0]['identity']
 require(str(server['pgid'])==capture['server_pgid'] and str(server['starttime'])==capture['server_start']
   and header['server_pgid']==capture['server_pgid'] and header['server_start']==capture['server_start']
   and header['session_id']==bound[0]['session_id']==capture['session_id'] and header['gpu_uuid']==bound[0]['gpu_uuid']==capture['gpu_uuid'],'cross-arm replay source/server substitution')
 require(bound[0]['t']<=spawn['t']<=capture['origin']<=capture['end']<=exit['t']<=spawn['deadline'],'actual bound replay clocks')
 cleanup=[r for r in events if r['event']=='cleanup' and r['pgid']==spawn['pgid']]
 final=[r for r in events if r['event']==label+'_cleanup_start']
 require(len(cleanup)==len(final)==1 and cleanup[0]['remaining']==[] and exit['t']<=cleanup[0]['t']<=final[0]['cleanup_start'],'replay cleanup before arm release')
 require(any(r['event']=='ready' and r['arm']==label and r['t']<=bound[0]['t'] for r in events),'health before actual replay binding')
 return identity

def reverse(root,config,events,capture):
 resource=candidate.final_resources(root,config,events,capture)
 release=read(root/'candidate-release.json');fresh=read(root/'control-admission.json')
 require(fresh['policy']=='FIRST_FRESH_NONHOST' and fresh['no_numeric_host_admission'] is True
   and fresh['original140_result']==fresh['sample']['probes']['cgroup']['eligible'],'experimental original140 report-only policy')
 admitted=[r for r in events if r['event']=='control_admitted'];dispatch=[r for r in events if r['event']=='spawn' and r.get('phase')=='control.prepare']
 require(len(admitted)==len(dispatch)==1 and all(admitted[0].get(k)==v for k,v in fresh.items()),'one sole fresh/second prepare')
 deadline=handoff.clocks(release,fresh['sample'],fresh['admitted_at'],dispatch[0]['t'],events[0]['work_deadline'],180)
 scope=read(root/'candidate-scope.json');sample=fresh['sample']
 value=candidate.verify_sample(root,sample,scope['binding'],capture['gpu_uuid'],deadline,allow_cpu=config['mode']=='CPU_FIXTURE_ONLY')
 require(value['eligible'] and candidate_resources.accepted(sample,deadline),'independent sole fresh nonhost')
 require(not any(r['event']=='host_boundary' and sample['end']<r['t']<dispatch[0]['t'] for r in events),'no host boundary between fresh and dispatch')
 require(scope['binding']['controller']['boot_id']==capture['boot_id'],'candidate release controller boot')
 return resource

def check(root,config,events,control,candidate_arm,hardware):
 root=Path(root)
 require(config['arm_order']==['candidate','control'],'explicit reverse order contract')
 require(hardware['name']=='NVIDIA RTX PRO 6000 Blackwell Server Edition' and hardware['memory_mib']>=90000 and hardware['driver'],'actual hardware')
 handoff.pair(candidate_arm['capture'],control['capture'])
 candidate.package(root,config,candidate_arm['capture'])
 scope=read(root/'candidate-scope.json');identity=scope['binding']['members'][str(scope['binding']['pgid'])]
 retained=[r for r in events if r['event']=='candidate_root_retained']
 require(len(retained)==1 and all(retained[0]['identity'][k]==identity[k] for k in ('pid','pgid','sid','uid','starttime','boot_id')),'candidate initial retained root lifetime')
 for label,value in (('candidate',candidate_arm),('control',control)):replay_join(root,events,label,value['capture'])
 require(candidate_arm['capture']['end']<=read(root/'candidate-release.json')['start'],'candidate replay before reverse cleanup')
 resource=reverse(root,config,events,candidate_arm['capture'])
 native=pair_resources.final(root,config,events,control['capture'])
 source=read(Path(__file__).with_name('control-plan.json'))
 for name,digest in source['installed_source_sha256'].items():
  require(hashlib.sha256((root/'control-package/installed-source'/name).read_bytes()).hexdigest()==digest,'actual installed shipping source '+name)
 host=candidate.host_evidence(root,config,events)
 require(not host['oom_increased'],'raw host OOM increase')
 labels=[r['label'] for r in map(json.loads,(root/'host-boundaries.jsonl').read_text().splitlines())]
 require(labels==LABELS,'complete reversed host boundary chronology')
 compatibility=read(root/'query-compatibility.json')
 require(set(compatibility)=={'gpu','apps'} and all(row['ok'] is True and row['cleanup_verified'] is True for row in compatibility.values()),'GPU query compatibility')
 def verify(value):
  if isinstance(value,dict):
   if {'path','sha256','bytes','complete'}<=value.keys():artifacts.verify(root/'query-artifacts',value,time.monotonic()+60)
   else:
    for child in value.values():verify(child)
  elif isinstance(value,list):
   for child in value:verify(child)
 verify(compatibility)
 return dict(reverse_resource=resource,final_native=native,host_evidence=host,owned_groups_reaped=True,experimental_nonhost_admission=True,original140_report_only=True)
