"""Reverse producer/consumer joins using accepted within-arm and raw resource checks."""
import csv,hashlib,json
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
 cfg=read(root.parent/'config.json')
 require(spawn['argv']==cfg[label]['replay']['argv'] and spawn['deadline']<=min(events[0]['work_deadline'],spawn['t']+cfg[label]['replay']['seconds']),'configured replay source/deadline')
 expected_argv=hashlib.sha256(b''.join(value.encode()+b'\0' for value in spawn['argv'])).hexdigest()
 require(identity['cmdline_sha256']==expected_argv and identity['sid']==identity['pid'],'actual replay command/session identity')
 header=json.loads((root/label/'events.jsonl').read_bytes().splitlines()[0])
 require(identity is not None and header['pid']==header['replay_pgid']==spawn['pid']==spawn['pgid']==exit['pid']==identity['pid']==identity['pgid']
   and identity['starttime']==header['replay_starttime'] and identity['boot_id']==header['boot_id']==capture['boot_id'],'actual replay PID/starttime/PGID/boot join')
 server=bound[0]['identity']
 require(all(identity[key]==server[key] for key in ('uid','cgroup','pid_namespace','boot_id')),'replay/server ownership scope')
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
 phase_deadlines(events)
 verified=[row for row in events if row['event']=='final_verification_pass']
 require(len(verified)==1 and verified[0]['start']<=verified[0]['t']<=verified[0]['deadline']<=min(events[0]['work_deadline'],verified[0]['start']+config['verify_seconds']),'final lifecycle verification allocation')
 handoff.pair(candidate_arm['capture'],control['capture'])
 candidate.package(root,config,candidate_arm['capture'])
 raw_hardware=list(csv.reader((root/'hardware.stdout').read_text().strip().splitlines()))
 require(len(raw_hardware)==1 and [x.strip() for x in raw_hardware[0]]==[hardware['name'],str(hardware['memory_mib']),hardware['driver'],hardware['uuid']],'actual raw hardware identity')
 for label in ('candidate','control'):
  health(root,events,config,label);preparation_hardware(root,label,hardware)
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
 transitions=[row for row in events if row['event']=='observer_transition']
 require(transitions and all(row['limit']==60 and 0<=row['spent']<=60 for row in transitions) and all(a['spent']<=b['spent'] for a,b in zip(transitions,transitions[1:])),'shared observer transition allocation')
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


def health(root,events,config,label):
 ready=[row for row in events if row['event']=='ready' and row['arm']==label]
 beginnings=[row for row in events if row['event']==('candidate_root_retained' if label=='candidate' else 'native_spawn')]
 require(len(ready)==len(beginnings)==1,'one bounded startup readiness')
 deadline=beginnings[0]['startup_deadline']
 require(beginnings[0]['t']<=ready[0]['t']<=deadline<=min(events[0]['work_deadline'],beginnings[0]['t']+config[label]['server']['seconds']),'original startup deadline')
 exits=[row for row in events if row['event']=='exit' and row.get('phase','').startswith(label+'.health-')]
 require(exits and [row['phase'] for row in exits]==[label+'.health-'+str(i) for i in range(len(exits))] and len(exits)<=int(config[label]['server']['seconds']/2)+1,'bounded sequential health probes')
 for index,row in enumerate(exits):
  stdout=(root/(row['phase']+'.stdout')).read_bytes();stderr=(root/(row['phase']+'.stderr')).read_bytes()
  require(stderr==b'' and row['t']<=ready[0]['t'],'raw health error/clock')
  if index==len(exits)-1:
   header,body=stdout.split(b'\n',1);meta=json.loads(header)
   require(row['returncode']==0 and meta['status']==200 and meta['body_limit']==65536 and len(body)<=65536,'actual first health200 raw body')
  elif row['returncode']==75:
   require(stdout.startswith(b'URLError(') and b'ConnectionRefusedError' in stdout,'actual refused-connection health receipt')
  else:
   import phase_tools
   require(phase_tools.startup_http_not_ready(row['returncode'],root/(row['phase']+'.stdout'),config[label]['ready_retry_http_statuses']),'pinned empty503 startup shape')
 return True


def preparation_hardware(root,label,hardware):
 folder=root/(label+'-package');gpu=read(folder/'actual-gpu-query.json')
 rows=list(csv.reader(gpu['raw'].strip().splitlines()))
 require(len(rows)==1 and [value.strip() for value in rows[0]]==[hardware['name'],str(hardware['memory_mib']),hardware['driver'],hardware['uuid']]
   and all(gpu[key]==hardware[key] for key in ('name','memory_mib','driver','uuid')),'same actual preparation GPU '+label)
 head=read(folder/'host-headroom.json')
 totals=[row.split(':',1)[1].split() for row in head['meminfo'].splitlines() if row.startswith('MemTotal:')]
 require(len(totals)==1 and len(totals[0])==2 and totals[0][0].isdigit() and totals[0][1]=='kB','measured total host capacity')
 total=int(totals[0][0])*1024;maximum=head['cgroup_memory_max']
 require(maximum=='max' or maximum.isdigit(),'measured preparation cgroup max')
 effective=min(total,int(maximum)) if maximum.isdigit() else total
 require(head['effective_memory_bytes']==effective and effective>=170*1024**3 and head['tmp_free_bytes']>=80*1024**3,'unchanged package total-capacity/tmp guards')
 # Capacity guard is not original140 available-memory eligibility or an OOM safety margin.
 return True


def phase_deadlines(events):
 for spawn in (row for row in events if row['event']=='spawn'):
  phase=spawn['phase']
  if phase=='observer' or phase.endswith('.server'):continue
  exits=[row for row in events if row['event']=='exit' and row.get('phase')==phase and row['pid']==spawn['pid']]
  # The producer invokes this checker in its sole pair.reduce child; final raw audit sees the completed child too.
  if phase=='pair.reduce' and not exits:continue
  cleaned=[row for row in events if row['event']=='cleanup' and row['pgid']==spawn['pgid']]
  require(len(exits)==len(cleaned)==1 and exits[0]['returncode']==0 and cleaned[0]['remaining']==[]
    and spawn['t']<=exits[0]['t']<=cleaned[0]['t']<=spawn['deadline'],'original command/health completion deadline '+phase)
 return True
