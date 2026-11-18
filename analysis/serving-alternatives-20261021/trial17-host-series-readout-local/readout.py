"""Read-only, 600-second bounded diagnosis of retained trial16/17 host samples."""
import base64, collections, hashlib, html, importlib.util, json, re, signal, statistics, sys, time, zlib
from pathlib import Path

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
OLD=BASE/'trial17-startup-readout-local/readout.py'
assert hashlib.sha256(OLD.read_bytes()).hexdigest()=='3d5bfa29e72bc48d7b3ab8a786330ca8d6b54e62bd69e364c6c76474c336b52a'
spec=importlib.util.spec_from_file_location('retained_readout',OLD)
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
raw,read=old.raw,old.read
G=1024**3

def pairs(s):return {k:int(v) for k,v in map(str.split,s.splitlines())}
def psi(s):return {a[0]:{k:float(v) if k!='total' else int(v) for k,v in (i.split('=') for i in a[1:])} for a in map(str.split,s.splitlines())}
def devices(s):return {a[0]:{k:int(v) for k,v in (i.split('=') for i in a[1:])} for a in map(str.split,s.splitlines())}
def status(s):return {k:v.strip() for k,v in (x.split(':',1) for x in s.splitlines())}

def load(root,path,expected):
 data=raw(path);assert hashlib.sha256(data).hexdigest()==expected
 receipt=read(root/'host-series.result.json');scope=read(root/'candidate-scope.json')
 assert receipt['pass_'] and receipt['error'] is None and receipt['bytes']==len(data) and receipt['series_sha256']==expected
 rows=[json.loads(x) for x in data.splitlines()];assert len(rows)==receipt['samples']
 stdout=raw(root/'candidate.server.stdout').decode()
 worker_ids=set(map(int,re.findall(r'\(Worker pid=(\d+)\)',stdout)));assert len(worker_ids)==1
 worker_pid=worker_ids.pop();worker_identity=scope['binding']['members'][str(worker_pid)]
 last=-1;identities=collections.Counter();cmdlines=collections.Counter();parents={};process_missing=0;ooms=[]
 for r in rows:
  assert r['run_id']==scope['run_id'] and r['scope']==scope['observer']
  assert receipt['start']<=r['start']<=r['end']<=receipt['end'] and r['start']>last;last=r['end']
  for fc in r['file_clocks'].values():assert r['start']<=fc['start']<=fc['end']<=r['end']
  events=pairs(r['raw']['events']);assert events==r['oom_events'];ooms.append(events)
  if 'events_local' in r['raw']:assert all(pairs(r['raw']['events_local'])[k]==0 for k in ('oom','oom_kill','oom_group_kill'))
  assert all(events[k]==0 for k in ('oom','oom_kill','oom_group_kill'))
  for io in r['io_diagnostics'].values():assert r['start']<=io['start']<=io['end']<=r['end']
  for p in r.get('processes',[]):
   if p['pid']!=worker_pid:continue
   if not p.get('identity') or p.get('gone_after_read'):process_missing+=1;continue
   # Diagnostic continuity only, not a signal/admission/scientific identity gate.
   # Both retained runs have two early samples with a different cmdline hash.
   # Preserve those variants; every other full identity field must match.
   assert {k:v for k,v in p['identity'].items() if k!='cmdline_sha256'}=={k:v for k,v in worker_identity.items() if k!='cmdline_sha256'}
   cmdlines[p['identity']['cmdline_sha256']]+=1
   assert r['start']<=p['start']<=p['end']<=r['end']
   assert int(status(p['status'])['PPid'])==p['ppid']
   identities[(p['pid'],p['identity']['starttime'],p['identity']['boot_id'],p['ppid'])]+=1
   observed=parents.setdefault(p['ppid'],dict(first_sample_start=r['start'],last_sample_end=r['end'],samples=0))
   observed['last_sample_end']=r['end'];observed['samples']+=1
 assert identities
 return rows,dict(bytes=len(data),sha256=expected,samples=len(rows),receipt=receipt,scope=scope,
  worker_pid=worker_pid,worker_identity=worker_identity,worker_observed_identities=[dict(pid=k[0],starttime=k[1],boot_id=k[2],ppid=k[3],samples=v) for k,v in identities.items()],
  worker_missing_or_raced_samples=process_missing,worker_cmdline_hash_variants=dict(cmdlines),worker_parent_observation_spans=parents,
  identity_policy='Diagnostic continuity: all identity fields except mutable cmdline hash must equal per-run scope; preserve all cmdline variants. Not signal/admission authority or a replacement scientific gate.',all_recorded_cgroup_OOM_counters_zero=True,
  process_sample_rows=sum(bool(r.get('processes')) for r in rows),io_sample_rows=sum(bool(r['io_diagnostics']) for r in rows),
  first_start=rows[0]['start'],last_end=rows[-1]['end'])

def summarize(rows,origin,span,worker):
 selected=[r for r in rows if origin<=r['start'] and r['end']<=origin+span];assert selected
 stats=[pairs(r['raw']['stat']) for r in selected]
 currents=[int(r['raw']['current'])/G for r in selected]
 counters={k:stats[-1][k]-stats[0][k] for k in ('pgscan','pgsteal','pgmajfault','workingset_refault_file')};assert min(counters.values())>=0
 io=[r['io_diagnostics'] for r in selected if r['io_diagnostics']];assert len(io)>1
 first,last=io[0],io[-1];a,b=psi(first['host_io_pressure']['raw']),psi(last['host_io_pressure']['raw'])
 interval=last['host_io_pressure']['start']-first['host_io_pressure']['start'];assert interval>0
 pressure={k:dict(total_delta_microseconds=b[k]['total']-a[k]['total'],fraction_of_counter_interval=(b[k]['total']-a[k]['total'])/1e6/interval) for k in ('some','full')}
 d0,d1=devices(first['cgroup_io']['raw']),devices(last['cgroup_io']['raw'])
 device_deltas={d:dict(first=d0.get(d),last=d1.get(d),delta={k:d1[d][k]-d0[d][k] for k in d1[d]} if d in d0 and d in d1 else None) for d in sorted(d0.keys()|d1.keys())}
 procs=[p for r in selected for p in r.get('processes',[]) if p['pid']==worker and p.get('identity') and not p.get('gone_after_read')]
 rss=[int(status(p['status'])['VmRSS'].split()[0])*1024/G for p in procs]
 return dict(requested_span_seconds=span,sample_count=len(selected),observed_sample_start_offset=selected[0]['start']-origin,observed_sample_end_offset=selected[-1]['end']-origin,
  cgroup_current_GiB=dict(min=min(currents),median=statistics.median(currents),observed_max=max(currents)),
  cgroup_stat_counter_deltas=counters,shmem_GiB_first_last=[stats[0]['shmem']/G,stats[-1]['shmem']/G],
  worker_state_sample_counts=dict(collections.Counter(p['state'] for p in procs)),worker_sample_count=len(procs),
  worker_RSS_GiB_first_last=[rss[0],rss[-1]] if rss else None,
  host_global_IO_PSI=dict(counter_interval_seconds=interval,first_counter_start_offset=first['host_io_pressure']['start']-origin,last_counter_start_offset=last['host_io_pressure']['start']-origin,**pressure),
  cgroup_block_device_deltas_UNSUMMED=device_deltas,
  device_counter_equality_not_topology_proof=[(a,b) for i,a in enumerate(sorted(d1)) for b in sorted(d1)[i+1:] if d1[a]==d1[b]])

def plot(rows_by_trial,traces,key,title,ymax,w=680):
 h=205;x0,y0=48,18;pw,ph=w-68,145
 out=[f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{html.escape(title)}"><title>{html.escape(title)}</title>',f'<rect x="{x0}" y="{y0}" width="{pw*600/1800}" height="{ph}" fill="#888" opacity=".12"/>']
 for y in (0,ymax/2,ymax):
  yy=y0+ph*(1-y/ymax);out.append(f'<path d="M{x0},{yy}H{x0+pw}" stroke="currentColor" opacity=".15"/><text x="42" y="{yy+4}" text-anchor="end">{y:g}</text>')
 for t in (0,600,1200,1800):out.append(f'<text x="{x0+pw*t/1800}" y="183" text-anchor="middle">{t}</text>')
 for trial,color in [('16','#329fcb'),('17','#e39439')]:
  tr=traces[trial];origin=tr['startup_origin'];end=tr['ready'][0]['t'] if tr['ready'] else tr['startup_deadline'];points=[];last=-100
  for r in rows_by_trial[trial]:
   t=r['start']-origin
   if t<0 or r['end']>end:continue
   if key=='memory':
    if t-last<10:continue
    value=int(r['raw']['current'])/G;last=t
   else:
    if not r['io_diagnostics']:continue
    value=psi(r['io_diagnostics']['host_io_pressure']['raw'])['some']['avg10']
   points.append(f'{x0+pw*t/1800:.2f},{y0+ph*(1-value/ymax):.2f}')
  out.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2"/>')
 out.append(f'<text x="{x0+pw/2}" y="201" text-anchor="middle">Seconds from startup origin</text></svg>')
 return ''.join(out)

def render(report,rows,traces,dest):
 a,b=[report['summaries'][i]['first600'] for i in ('16','17')]
 def cell(s,key):
  if key=='memory':return f"{s['cgroup_current_GiB']['median']:.2f} / {s['cgroup_current_GiB']['observed_max']:.2f} GiB"
  if key=='psi':return f"{100*s['host_global_IO_PSI']['some']['fraction_of_counter_interval']:.2f}% ({s['host_global_IO_PSI']['counter_interval_seconds']:.2f} s)"
  if key=='states':return ', '.join(f'{k}: {v}' for k,v in s['worker_state_sample_counts'].items())
  return f"{s['cgroup_stat_counter_deltas'][key]:,}"
 table=''.join('<tr><td>'+label+'</td><td>'+cell(a,key)+'</td><td>'+cell(b,key)+'</td></tr>' for label,key in [('Cgroup current: median / observed max','memory'),('Host-global I/O PSI some: counter-time share','psi'),('Worker main-task state samples','states'),('Cgroup page scans Δ','pgscan'),('Cgroup major faults Δ','pgmajfault')])
 page='''<!doctype html><html><head><meta charset="utf-8"><style>body{font:14px/1.5 var(--font-sans,system-ui);color:var(--foreground);margin:0}h2{font-size:17px;margin:22px 0 8px}p{margin:0 0 12px}table{width:100%;border-collapse:collapse;font-size:13px}td,th{text-align:left;vertical-align:top;padding:8px 5px;border-bottom:1px solid var(--border,#888)}svg{width:100%;height:auto;display:block}svg text{fill:currentColor;font:11px system-ui}code{font:12px monospace;overflow-wrap:anywhere}.muted{color:var(--muted-foreground)}.blue{color:#329fcb}.orange{color:#e39439}summary{cursor:pointer}li{margin-bottom:8px}</style></head><body>
 <p><strong>Richard Hamming — Resource wait signatures appear in both runs; the loading mechanism remains unidentified.</strong> Trial 16 reached readiness after 972.793 s. Trial 17 exhausted its original 1,800 s startup while loading remained incomplete. This recovered file sharpens the resource evidence; it does not establish a memory, NFS, CPU or order cause.</p>
 <h2>Same elapsed interval, different sessions</h2><p>First 600 s after each supervisor startup origin; both are before main checkpoint completion. This exploratory window avoids including trial 16’s later completed-loader stage. Equal elapsed time still does not match shard progress or environmental history.</p>
 <table><thead><tr><th>Recorded evidence</th><th>Trial 16: ready</th><th>Trial 17: timeout</th></tr></thead><tbody>TABLE</tbody></table>
 <p class="muted">Memory uses all 0.5 s samples wholly inside the window. Worker states are approximately 5 s snapshots after worker creation, not percentages of blocked wall time. D is sampled uninterruptible wait, without a recorded wait channel. PSI uses the actual approximately 591 s between I/O snapshots.</p>
 <h2>Startup trajectories</h2><p><span class="blue">Blue: trial 16</span> · <span class="orange">Orange: trial 17</span> · shaded: common 600 s. Each curve stops at its own readiness or startup deadline.</p><p>Cgroup memory.current (GiB)</p>MEMORY<p>Host-global I/O PSI some avg10 (%)</p>PSI
 <p class="muted">Memory curves show one retained sample per approximately 10 s; table maxima use every eligible sample. PSI plots every available approximately 10 s observation. No interpolation across sessions, elapsed-time effect estimate or threshold fit.</p>
 <h2>What changed in the evidence</h2><p>All 4,229 recovered trial 17 samples join the primary OOM projection byte values and clocks. Every recorded cgroup OOM counter is zero. Worker identity joins within each run include PID, starttime, boot, parent, process group and namespace. The same numeric PID across sessions is not treated as the same process.</p>
 <p>Both runs show high host-global I/O pressure and frequent sampled worker D states. In the common interval, successful trial 16 has greater recorded cgroup current, page scans and major faults. These summaries do not support the simple claim that failed trial 17 had greater measured memory pressure; they also cannot exclude a memory-related mechanism. Later intervals mix different loading stages and exposure lengths.</p>
 <h2>Limits that still decide the interpretation</h2><ul><li><code>/proc/pressure/io</code> is host-global. Bound-cgroup <code>io.stat</code> is block-device accounting, not worker attribution or NFS checkpoint service. Device counters are retained separately; equal or stacked-device counters are not summed.</li><li>RSS contains shared pages; shmem overlaps file accounting. Neither is added into a private footprint. Cumulative memory.peak and sampled current do not establish a safe margin.</li><li>No worker CPU ticks/utilization, thread-level wait channels, per-tensor timings, NFS latency/cache history or contention intervention are recorded. Slow loading itself can alter cache/reclaim/wait signatures. One successful and one failed candidate-first run cannot identify an order effect.</li></ul>
 <p><strong>Decision:</strong> retain “startup deadline exhausted during incomplete loading; mechanism unidentified.” No timeout, loader, cache, order, threshold or successor change follows from these observations. Slot remains released; missing scientific dependencies and ratio null remain.</p>
 <details><summary>Collection and reproducibility</summary><p>One GET, 27,510,319 B, exact pinned SHA <code>9b04bd666d43476cdb4d8cc3bf23c769da93d5d68db29e86b54fb55416246432</code>; 50.649 s collector, exit 0, no retry. Exact terminal/source/quota readbacks before and after. Original zero-API failed outer invocation remains unchanged. <a href="READOUT.json">READOUT.json</a> contains raw input pins, 600 s / 900 s / own-startup summaries, unsummed device deltas, clocks, identity and limitations. <a href="../readout.py">readout.py</a> is the separate CPU-only analysis. This is not full raw scientific qualification.</p></details></body></html>'''
 def responsive(key,title,ymax):return '<div class="desktop">'+plot(rows,traces,key,title,ymax)+'</div><div class="mobile">'+plot(rows,traces,key,title,ymax,340)+'</div>'
 page=page.replace('height:auto;display:block','height:205px;display:block').replace('</style>','.mobile{display:none}@media(max-width:480px){.desktop{display:none}.mobile{display:block}}</style>')
 page=page.replace('TABLE',table).replace('MEMORY',responsive('memory','Bound cgroup memory.current',180)).replace('PSI\n',responsive('psi','Host-global I/O PSI some avg10',100)+'\n')
 page=page.replace('The same numeric PID across sessions is not treated as the same process.','The same numeric PID across sessions is not treated as the same process. Both runs have two early worker command-line hash variants; these are retained explicitly, while every other identity field matches the within-run scope. This diagnostic continuity join grants no signal, admission or scientific qualification authority.')
 page=page.replace('Worker identity joins within each run include PID, starttime, boot, parent, process group and namespace.','Worker continuity joins within each run use PID, starttime, boot, process group and namespace; parent observations are retained separately.')
 page=page.replace('This diagnostic continuity join grants no signal, admission or scientific qualification authority.','This diagnostic continuity join grants no signal, admission or scientific qualification authority. One trial 17 worker snapshot at 2908.184 s shows parent PID 1 after cleanup began at 2907.180 s; it is outside every startup summary, not evidence of a readiness or spawn failure.')
 (dest/'READOUT.html').write_text(page)

def main(destination):
 signal.alarm(600);started=time.monotonic();dest=Path(destination).resolve();dest.mkdir(exist_ok=False);forbidden=[]
 def guard(event,args):
  if event in ('socket.connect','subprocess.Popen','os.system'):forbidden.append(event);raise RuntimeError('network/process execution forbidden')
  if event=='open' and isinstance(args[0],str) and args[0].startswith(('/dev/nvidia','/dev/dri','/proc/driver/nvidia')):forbidden.append(event);raise RuntimeError('GPU access forbidden')
 sys.addaudithook(guard);raw(__file__);raw(OLD)
 prior=read(BASE/'trial17-startup-readout-local/final/READOUT.json')
 assert hashlib.sha256(raw(BASE/'trial17-startup-readout-local/final/READOUT.json')).hexdigest()=='0240b26309f3c2a327cd4ea3f2f27976d8ad37685ac3f2df562de6eb78e3efd0'
 for path,digest in prior['inputs_sha256'].items():assert hashlib.sha256(raw(path)).hexdigest()==digest,path
 inv=BASE/'trial17-host-series-invocation-r2-20261024';exit=read(inv/'EXIT.json')
 for file,key in [('stdout','stdout_sha256'),('stderr','stderr_sha256'),('direct.py','source_sha256'),('DIRECT-COMPLETION.json','completion_sha256'),('FUNCTION-RETURNED.json','returned_sha256')]:assert hashlib.sha256(raw(inv/file)).hexdigest()==exit[key]
 collection=BASE/'trial17-host-series-collected-20261024/COLLECTION.json';assert hashlib.sha256(raw(collection)).hexdigest()==exit['collection_sha256']
 assert read(collection)==read(inv/'FUNCTION-RETURNED.json')
 assert exit['observed_exit_code']==0 and exit['stderr_bytes']==0 and exit['GET_attempts']==1 and exit['retry_attempts']==0
 assert all(exit[k]['source_exact'] and exit[k]['kernel_id']==140530941 and exit[k]['version']==1 and exit[k]['status']=='ERROR' and exit[k]['wire_seconds']['reserved_s']=='0' for k in ('before','after'))
 assert exit['before']['wire_seconds']==exit['after']['wire_seconds']
 for file,digest in [('trial17-one-host-series-remote-approved.json','52911eee6d99e19009140b824f44718f7c8795dac7f8f016ccb7088975476b92'),('trial17-host-series-replacement-invocation-approved.json','8e92ccf83503160b05949082c4656293d3c007be9fc03c98a5f335812766d4d5'),('trial17-one-host-series-manifest.json','37717363767bb66d7010d06e56277710465bfcb82e7e73999167c658b2d812a6')]:assert hashlib.sha256(raw(BASE/'coordination-20261024'/file)).hexdigest()==digest
 rows={};metadata={};traces={};summaries={}
 for trial,root,path,digest in [('16',old.R16,old.R16/'host-series.jsonl','1c081e8050d9388a8871bc98f923861e034a32d4aa8655c063ae1a1076cd1d1c'),('17',old.R17,BASE/'trial17-host-series-collected-20261024/members/serving-candidate-first-capacity17/output/host-series.jsonl','9b04bd666d43476cdb4d8cc3bf23c769da93d5d68db29e86b54fb55416246432')]:
  rows[trial],metadata[trial]=load(root,path,digest)
  _,traces[trial]=old.trace(root);t=traces[trial];origin=t['startup_origin'];span=t['ready_after_startup_origin_s'] if t['ready'] else t['startup_deadline']-origin
  summaries[trial]={name:summarize(rows[trial],origin,duration,metadata[trial]['worker_pid']) for name,duration in [('first600',600),('first900',900),('own_startup',span)]}
 envelope=read(old.P17/'remote-readback/primary-current-once/PRIMARY.json');bundle=json.loads(zlib.decompress(base64.b64decode(envelope['compressed_base64'],validate=True)))
 projection=[dict(start=r['start'],end=r['end'],events_raw=r['raw']['events'],oom_events=r['oom_events']) for r in rows['17']]
 assert projection==bundle['oom_projection']
 assert metadata['17']['bytes']==27510319 and metadata['17']['samples']==4229
 assert raw(old.P17/'host_observer.py')==raw(old.P16/'host_observer.py')
 report=dict(status='LOCAL_RESOURCE_READOUT_MECHANISM_UNIDENTIFIED',collection=exit,raw_metadata=metadata,traces=traces,summaries=summaries,
  primary_OOM_projection_exact_join=True,prior_readout_sha256='0240b26309f3c2a327cd4ea3f2f27976d8ad37685ac3f2df562de6eb78e3efd0',
  window_policy='Exploratory common first600s avoids completed main-loader stage in16. First900 and ownstartup retained as sensitivity/context. Equal elapsed time does not match shard stage; unequal fullstartup exposures disclosed. No threshold or scientific gate fitted.',
  clock_policy='Use supervisor monotonic startup origin per run; require whole sample inside window. PSI denominator uses actual first/last nested IO capture starts; sequential/non-atomic reads retained.',
  causal_structure=['environment/session/cache-history -> resource service and loading progress','loading progress at t -> memory/cache/reclaim/wait signatures at t+1','resource pressure at t -> possible loading progress at t+1','source/package held observationally equal; environment/history and semantic loader stages are not controlled'],
  conclusion='Startup1800 exhausted during incomplete loading. Resource wait signatures occur in both successful16 and failed17; observed current/reclaim in common window are higher in16. Mechanism remains unidentified.',
  limitations=['Host-global PSI cannot be attributed to candidate or cgroup.','Cgroup io.stat block devices not summed: aliases/stack topology and NFS service are not identified. Missing endpoint device delta is null.','Worker main-task D samples are not continuous blocked time, thread-wide states, CPU utilization or NFS-specific waits.','Process RSS includes shared pages. file includes shmem. No sum into private footprint; cumulative peak not reset and observed maxima do not establish safety margin.','Cgroup scope includes other in-scope processes; sampled memory.current is not incremental candidate memory.','No per-worker CPU ticks, wait channels, per-tensor service time, NFS client/server/cache telemetry or controlled intervention.','One success/failure pair in candidate-first order cannot identify order effect or general reliability.'],
  decision='Retain incomplete-loading startup exhaustion and unidentified mechanism. No runtime/loader/cache/order/threshold/timeout change, additional fetch or successor packet.',
  preserved_local_analysis_failure='development-r1: full identity equality rejected mutable cmdline hash; two early samples differ in each run, all other fields exact. Source/stdout/stderr retained; analysis-only correction, no scientific contract change.',
  ratio=None,full_scientific_qualification=False,slot='RELEASED',network_process_GPU_attempts=forbidden)
 for path,digest in list(old.INPUTS.items()):assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
 report.update(inputs_sha256=dict(old.INPUTS),elapsed_seconds=time.monotonic()-started)
 (dest/'READOUT.json').write_text(json.dumps(report,indent=2)+'\n');render(report,rows,traces,dest)
 print(json.dumps(dict(status=report['status'],inputs=len(old.INPUTS),seconds=time.monotonic()-started,projection_samples=len(projection),forbidden_attempts=forbidden,first600={t:summaries[t]['first600'] for t in ('16','17')})))
if __name__=='__main__':
 if len(sys.argv)!=2:raise SystemExit('readout.py FRESH_ABSOLUTE_OUTPUT')
 main(sys.argv[1])
