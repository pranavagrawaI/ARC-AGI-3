"""600s CPU-only retained-byte diagnosis. No model imports, API, edits or signals."""
import base64,collections,hashlib,html,json,signal,sys,time,zlib
from pathlib import Path

HERE=Path(__file__).resolve().parent;BASE=HERE.parent
P17=BASE/'trial-17-candidate-first-pair-r2-local'
P16=BASE/'trial-16-candidate-s10-r4-operational-local'
R17=P17/'remote-readback/primary-material'
R16=BASE/'trial16-three-pass-audit-result-20261024/evidence/serving-candidate-s10-16/output'
OVERLAY=BASE.parent/'m2-winners-20261001/audit/lord-runtime/overlay'
INPUTS={}
def raw(path):
 path=Path(path).resolve();data=path.read_bytes();digest=hashlib.sha256(data).hexdigest()
 if str(path) in INPUTS:assert INPUTS[str(path)]==digest,path
 INPUTS[str(path)]=digest;return data
def read(path):return json.loads(raw(path))
def normalize(value):
 if isinstance(value,str):
  return value.replace('/kaggle/working/serving-candidate-s10-16','/kaggle/working/WORKING').replace('/kaggle/working/serving-candidate-first-capacity17','/kaggle/working/WORKING')
 if isinstance(value,list):return list(map(normalize,value))
 if isinstance(value,dict):return {k:normalize(v) for k,v in value.items()}
 return value
def progress(root):
 err=raw(root/'candidate.server.stderr').decode();out=raw(root/'candidate.server.stdout').decode()
 rows=[]
 for m in __import__('re').finditer(r'(\d+)/97 \[([0-9:]+)[<,]',err):
  elapsed=0
  for number in m[2].split(':'):elapsed=elapsed*60+int(number)
  rows.append(dict(shards=int(m[1]),loader_elapsed_s=elapsed))
 ram=__import__('re').search(r'Checkpoint size: ([\d.]+) GiB\. Available RAM: ([\d.]+) GiB',out)
 terms=('Filesystem type for checkpoints','Skipping auto-prefetch','Initialized PLE embedding','Loading weights took','Model loading took','Multi-modal warmup completed','Loading model from scratch')
 stages=[dict(line=i,text=line) for i,line in enumerate(out.splitlines(),1) if any(s in line for s in terms)]
 return dict(main97_progress=rows,first21=next(r for r in rows if r['shards']==21),last=rows[-1],
  full97=any(r['shards']==97 for r in rows),checkpoint_GiB=float(ram[1]),logged_available_RAM_GiB=float(ram[2]),
  auto_prefetch_skipped='Skipping auto-prefetch' in out,PLE_BF16_CPU_pinned='weight_dtype=torch.bfloat16, weight_device=cpu, pinned=True' in out,
  stages=stages,fatal_log_lines=[line for line in (out+'\n'+err).splitlines() if any(t in line for t in ('Traceback','CUDA out of memory','OutOfMemoryError','ERROR '))])
def trace(root):
 rows=[json.loads(line) for line in raw(root/'supervisor.jsonl').splitlines()]
 retained=next(r for r in rows if r['event']=='candidate_root_retained')
 spawns=[r for r in rows if r['event']=='spawn' and r.get('phase','').startswith('candidate.health-')]
 exits=[r for r in rows if r['event']=='exit' and r.get('phase','').startswith('candidate.health-')]
 cleanup=[]
 for s in spawns:
  e=[r for r in exits if r['phase']==s['phase'] and r['pid']==s['pid']]
  c=[r for r in rows if r['event']=='cleanup' and r['pgid']==s['pgid']]
  assert len(e)==len(c)==1 and c[0]['remaining']==[] and s['t']<=e[0]['t']<=c[0]['t']<=s['deadline']
  cleanup.append(c[0])
 ready=[r for r in rows if r['event']=='ready' and r['arm']=='candidate']
 phase=[]
 for s in rows:
  if s['event']=='spawn' and s.get('phase') in ('corpus.prepare','candidate.prepare','candidate.identity'):
   e=next(r for r in rows if r['event']=='exit' and r['phase']==s['phase'] and r['pid']==s['pid'])
   phase.append(dict(phase=s['phase'],spawn=s['t'],exit=e['t'],elapsed_s=e['t']-s['t'],returncode=e['returncode']))
 begin=next(r for r in rows if r['event']=='candidate_cleanup_start')
 return rows,dict(first_event_t=rows[0]['t'],work_deadline=rows[0]['work_deadline'],server=next(r for r in rows if r['event']=='spawn' and r.get('phase')=='candidate.server'),
  retained=retained,startup_origin=retained['startup_deadline']-1800,startup_deadline=retained['startup_deadline'],ready=ready,
  ready_after_startup_origin_s=ready[0]['t']-(retained['startup_deadline']-1800) if ready else None,
  health_count=len(spawns),health_exit_counts=dict(collections.Counter(r['returncode'] for r in exits)),health_lifecycle_checks=True,
  first_health=spawns[0],last_health=spawns[-1],last_health_exit=exits[-1],last_health_cleanup=cleanup[-1],candidate_cleanup_start=begin,
  preceding_cleanup_events=rows[rows.index(begin)-2:rows.index(begin)+1],preparation_phases=phase)
def host(root):
 boundaries=[]
 for line in raw(root/'host-boundaries.jsonl').splitlines():
  r=json.loads(line);events={k:int(v) for k,v in (s.split() for s in r['raw']['events'].splitlines())}
  assert events==r['oom_events']
  boundaries.append(dict(label=r['label'],start=r['start'],end=r['end'],current_bytes=int(r['raw']['current']),max_bytes=int(r['raw']['maximum']),
   original140_computed_bytes=r['original140']['computed_bytes'],original140_eligible=r['original140']['eligible'],oom=events,io_diagnostics_present=bool(r.get('io_diagnostics'))))
 return dict(boundaries=boundaries,series_receipt=read(root/'host-series.result.json'))
def main(destination):
 signal.alarm(600);started=time.monotonic();dest=Path(destination).resolve();dest.mkdir(exist_ok=False);forbidden=[]
 def guard(event,args):
  if event=='socket.connect' or event=='subprocess.Popen':forbidden.append(event);raise RuntimeError('readout forbids network/process execution')
  if event=='open' and isinstance(args[0],(str,bytes)) and str(args[0]).startswith(('/dev/nvidia','/dev/dri','/proc/driver/nvidia')):forbidden.append(event);raise RuntimeError('readout forbids GPU access')
 sys.addaudithook(guard);raw(__file__)
 assert hashlib.sha256(raw(P17/'freeze.json')).hexdigest()=='34befabf21e93d7753e251ae6aa3e1131ab224f4395779f60f4845bf9a1f32aa'
 assert hashlib.sha256(raw(P16/'freeze.json')).hexdigest()=='b51120fedef231e6b5fdecd8882f03bb78653e4e221fa9b256cdd7d908c2ed5b'
 closure=BASE/'coordination-20261024/trial17-attempt-closed.json';assert hashlib.sha256(raw(closure)).hexdigest()=='8313f3c1020bf2311b70bbc136811f65d26b346ceeab11b4b76d1dbcb211ef53'
 primary=raw(P17/'remote-readback/primary-current-once/PRIMARY.json');assert hashlib.sha256(primary).hexdigest()==read(closure)['primary_sha256']
 envelope=json.loads(primary);decoded=zlib.decompress(base64.b64decode(envelope['compressed_base64'],validate=True))
 assert len(decoded)==envelope['decoded_bytes'] and hashlib.sha256(decoded).hexdigest()==envelope['decoded_sha256'];bundle=json.loads(decoded)
 for name,item in bundle['files'].items():
  data=raw(R17/name);assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256']
 rows17,t17=trace(R17);rows16,t16=trace(R16);logs17=progress(R17);logs16=progress(R16)
 health=bundle['health_inventory'];known=raw(R16/'candidate.health-0.stdout');known_sha=hashlib.sha256(known).hexdigest()
 outs=[r for r in health if r['path'].endswith('.stdout')];errs=[r for r in health if r['path'].endswith('.stderr')]
 assert len(outs)==len(errs)==t17['health_count']==873 and all(r['sha256']==known_sha and r['bytes']==len(known) for r in outs)
 assert {r['path'] for r in health}=={f'candidate.health-{i}.{suffix}' for i in range(873) for suffix in ('stdout','stderr')}
 assert all(r['bytes']==0 and r['sha256']==hashlib.sha256(b'').hexdigest() for r in errs)
 equal={}
 for name in ('candidate-launch.json','prepared.json','installed-source-identity.json','model-source-identity.json'):
  equal[name]=normalize(read(R16/'candidate-package'/name))==normalize(read(R17/'candidate-package'/name));assert equal[name]
 cfg17=read(P17/'packet-spec.json')['config'];cfg16=read(P16/'packet-spec.json')['config'];assert normalize(cfg17['candidate'])==normalize(cfg16['candidate'])
 identical=[]
 for name in ('candidate_prepare.py','candidate_library.py','candidate_runner.py','host_observer.py','http_once.py'):
  assert raw(P17/name)==raw(P16/name);identical.append(name)
 for name in ('phase_tools.py','pair_runner.py','pair_evidence.py','primary_bundle.py'):raw(P17/name)
 ids=read(R17/'candidate-package/installed-source-identity.json');source_refs={}
 for name in ('vllm/model_executor/model_loader/weight_utils.py','vllm/model_executor/model_loader/default_loader.py','vllm/models/qwen4_exp/nvidia/ngram_embedding.py'):
  data=raw(OVERLAY/name);assert hashlib.sha256(data).hexdigest()==ids[name];source_refs[name]=ids[name]
 host17=host(R17);host16=host(R16);projection=bundle['oom_projection']
 for r in projection:
  assert {k:int(v) for k,v in (s.split() for s in r['events_raw'].splitlines())}==r['oom_events']
 assert all(r['oom_events']['oom']==r['oom_events']['oom_kill']==r['oom_events'].get('oom_group_kill',0)==0 for r in projection)
 gpu17=read(R17/'candidate-package/actual-gpu-query.json');gpu16=read(R16/'candidate-package/actual-gpu-query.json')
 for root in (R17,R16):read(root/'candidate-package/host-headroom.json')
 result=read(R17/'result.json');raw(R17/'candidate-scope.json');raw(R17/'candidate-release.json')
 read(P17/'remote-readback/TERMINAL-HANDOFF.json')
 series=host17['series_receipt'];assert series['samples']==len(projection)==4229
 deadline=t17['startup_deadline'];bound=t17['preceding_cleanup_events'][0]['t'];assert bound>deadline
 assert t17['last_health_cleanup']['t']<deadline<t17['candidate_cleanup_start']['t']<t17['work_deadline']
 report=dict(status='LOCAL_STARTUP_READOUT_MECHANISM_UNIDENTIFIED',trial17=t17,trial16=t16,logs17=logs17,logs16=logs16,
  exact_setup=dict(normalized_actual_records_equal=equal,candidate_config_equal_after_only_working_path_normalization=True,identical_producer_sources=identical,
   normalization=['/kaggle/working/serving-candidate-s10-16','/kaggle/working/serving-candidate-first-capacity17'],installed_source_pins_equal=len(ids),loader_source_pins=source_refs,
   full_weight_hash_verified=read(R17/'candidate-package/model-source-identity.json')['full_weight_sha256_verified'],gpu17=gpu17,gpu16=gpu16),
  deadline_reconstruction=dict(startup_origin=t17['startup_origin'],startup_deadline=deadline,last_health_exit=t17['last_health_exit']['t'],last_health_cleanup=t17['last_health_cleanup']['t'],
   remaining_after_last_health_cleanup_s=deadline-t17['last_health_cleanup']['t'],next_unissued_health='candidate.health-873',
   inferred_error_clock_bracket=[deadline,bound],error_clock_not_directly_recorded=True,cleanup_start=t17['candidate_cleanup_start']['t'],cleanup_after_deadline_s=t17['candidate_cleanup_start']['t']-deadline,
   work_time_remaining_at_startup_deadline_s=t17['work_deadline']-deadline,
   source_supported_path='ready: successful cleanup of refused health872 -> sleep clipped to original startup end -> next loop checks parent alive -> start(next health) rejects expired end before opening logs/Popen. Generic spawn message is exhaustion detection, not evidence of fork failure.'),
  health_index=dict(stdout_entries=len(outs),stderr_entries=len(errs),all_stdout_equal_retained_trial16_refusal_sha256=known_sha,stdout_bytes_each=len(known),stderr_empty=True,
   provenance='Hash/length joins to retained actual16 bytes; raw17 health files were not downloaded or fabricated.'),
  host17=host17,host16=host16,available_OOM_evidence=dict(primary_projection_samples=len(projection),boundary_samples=len(host17['boundaries']),all_observed_oom_counters_zero=True,
   full_series_available_locally=False,series_bytes=series['bytes'],series_sha256=series['series_sha256'],independent_full_raw_qualification=False),
  first_failed_boundary='Candidate failed to reach first health200 by its original1800s startup end, while main97-shard loading remained incomplete. Producer stopped before replay/reverse admission/control.',
  interpretations=['Observed slow incomplete loading, not a demonstrated deadlock: completions continued to21/97; final repeated21 entry is not new progress.',
   'Same pinned source/model/prepared/launch semantics after working-path-only normalization; same BF16 CPU pinned PLE and skipped auto-prefetch branch. GPU class/VRAM/driver match; UUID, boot and session differ.',
   'Logged9.95GiB versus10.46GiB uses min(host available,cgroup limit-current), without inactive_file credit; it is not original140 or a memory-cause proof.',
   'Iterator progress spans safe_open/get_tensor/yield and model consumer work; it is not isolated NFS throughput. No per-worker CPU time, per-tensor/IO service timing or causal intervention is available.',
   'No fatal log lines observed; parent retained identity and cleanup binding support liveness, not worker health or successful loading. Full untruncated logs narrow but do not eliminate hidden/propagation failures.',
   'Both candidate-first trials include standalone16 success and17 failure; order alone cannot explain the contrast. No readiness probability, historical-rate gate, capacity ratio or safe peak margin is inferred.'],
  decisive_missing_discriminants=['Trial17 full host-series raw cgroup/stat/process-status and low-cadence hostIO pressure/cgroupIO records. Only hash/size, OOM projection and ten boundary samples are currently local.',
   'Per-worker CPU utilization, per-tensor CPU/copy/quantization service time, NFS client/server latency/cache history and node contention; existing series cannot fully supply these.',
   'Repeated or intervened evidence isolating environmental variation; one success/failure pair cannot identify a mechanism.'],
  one_decision_proposal=dict(action='Seek separate authority for one existing host-series.jsonl retrieval only; no GPU rerun or runtime changes.',
   remote_name='serving-candidate-first-capacity17/output/host-series.jsonl',expected_bytes=series['bytes'],expected_sha256=series['series_sha256'],proposed_byte_cap=28*1024**2,proposed_attempts=1,
   purpose='Compare startup memory/reclaim/process status and IO-pressure trajectories against retained16, separating supportive resource-pressure evidence from continued unidentified variation.',
   outcomes=['Concordant pressure signatures justify a focused mechanism hypothesis, not a causal conclusion or threshold/timeout change.','Absent/flat/unavailable signatures leave mechanism unresolved; no inference that IO/CPU/memory is safe or excluded.'],
   stop_rule='One exact existing file only after new approval; hash/size/identity/transport failure stops inconclusive. No residual pass, automatic rerun or additional collection.',currently_authorized=False),
  result=result,ratio=None,no_runtime_edits=True,no_GPU=True,no_external_network=True,no_successor_packet=True,forbidden_attempts=forbidden)
 for path,digest in list(INPUTS.items()):assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
 report.update(inputs_sha256=dict(INPUTS),elapsed_seconds=time.monotonic()-started)
 (dest/'READOUT.json').write_text(json.dumps(report,indent=2)+'\n');render(report,dest)
 print(json.dumps(dict(status=report['status'],inputs=len(INPUTS),seconds=report['elapsed_seconds'],trial17_last_progress=logs17['last'],trial16_first21=logs16['first21'],deadline=deadline,forbidden_attempts=forbidden)))

def render(r,dest):
 h=html.escape;t=r['deadline_reconstruction'];a=r['trial16'];b=r['trial17'];logs16=r['logs16'];logs17=r['logs17']
 rows=[('First21/97 loader elapsed','3m54s','27m47s'),('Main checkpoint progress','97/97 by12m50s','21/97; final repeat28m39s'),('First health200 after startup origin',f"{a['ready_after_startup_origin_s']:.3f}s",'Absent within1800s'),('Health attempts','472 refusal75 +1 success0','873 refusal75'),('Loader logged RAM','10.46GiB','9.95GiB'),('Auto-prefetch','Skipped','Skipped')]
 table=''.join('<tr>'+''.join('<td>'+h(str(v))+'</td>' for v in row)+'</tr>' for row in rows)
 text=f'''<!doctype html><html><head><meta charset="utf-8"><style>body{{margin:0;color:var(--foreground);font:14px/1.5 var(--font-sans,system-ui)}}p{{margin:0 0 12px}}h2{{font-size:17px;margin:20px 0 8px}}table{{width:100%;border-collapse:collapse;font-size:13px}}td,th{{text-align:left;vertical-align:top;padding:8px 5px;border-bottom:1px solid var(--border,#888)}}td:first-child{{width:38%}}code{{font:12px var(--font-mono,monospace);overflow-wrap:anywhere}}.note{{color:var(--muted-foreground)}}details{{margin:12px 0}}summary{{cursor:pointer}}li{{margin-bottom:6px}}a{{color:var(--primary)}}</style></head><body>
 <p><strong>Richard Hamming — Trial17 exhausted startup while still loading.</strong> The source explains the generic “deadline before spawn” as the next health probe’s guard. It does not identify why loading was slow. Local retained-byte diagnosis only; no new API, GPU or source edits.</p>
 <table><thead><tr><th>Evidence</th><th>Successful16</th><th>Failed17</th></tr></thead><tbody>{table}</tbody></table>
 <p class="note">Loader times are the main97-shard tqdm clock, separate from supervisor monotonic time. Progress includes tensor materialization and consumer work; it is not an NFS bandwidth measure. Sessions and GPU UUIDs differ.</p>
 <h2>The failure boundary</h2><p>Startup origin <code>{t['startup_origin']:.9f}</code>; original end <code>{t['startup_deadline']:.9f}</code>. Health872 cleanup completed at <code>{t['last_health_cleanup']:.9f}</code>, with{t['remaining_after_last_health_cleanup_s']:.6f}s left. The clipped sleep consumed that remainder; the next unissued probe was health873. Cleanup began{t['cleanup_after_deadline_s']:.6f}s after startup end, while{t['work_time_remaining_at_startup_deadline_s']:.3f}s of work budget remained. The exact raise timestamp is unrecorded; this is a source-supported path, not a remote stack trace.</p>
 <h2>What the comparison supports</h2><p>All39 installed source pins and model identities agree. Candidate config, preparation, flags, loader/cache settings and actual launch records agree after working-path normalization. Both used BF16 pinned CPU PLE and skipped auto-prefetch. The17 health index joins all873 stdout hashes to actual16 refusal bytes; this did not require fetching or inventing17 raw health files.</p>
 <p>Available17 OOM projection:4229 entries plus10 boundary samples, all observed OOM counters zero. Full raw host series is absent locally; this is not full resource qualification or a safe memory margin. Logged RAM differs, but neither RAM nor NFS is established as the cause. Parent liveness and untruncated logs do not prove every worker remained healthy.</p>
 <h2>One decision proposal</h2><p>Seek separate authority to retrieve only the existing <code>host-series.jsonl</code>:27,510,319B, pinned hash <code>9b04bd666d43476cdb4d8cc3bf23c769da93d5d68db29e86b54fb55416246432</code>; one attempt,28MiB cap. Its already-recorded memory/reclaim/process-status and low-cadence IO evidence may support a resource-pressure hypothesis. It cannot supply per-worker CPU time, per-tensor latency or NFS service causality. Stop on failure or mismatch; flat/missing signals leave the mechanism unidentified. No GPU rerun or runtime change follows automatically. <strong>This retrieval is not currently authorized.</strong></p>
 <p class="note">Candidate-first has both16 success and17 failure; order alone is not an established explanation. Slot released; FEASIBILITY_ONLY/ratio null and missing18 remain. No timeout, loader, cache, order, threshold, instrumentation or successor packet change.</p>
 <details><summary>Source and evidence map</summary><p><a href="READOUT.json">READOUT.json</a> pins every consumed source/artifact and contains exact events, progress, host boundaries, missing discriminants and proposal. <a href="readout.py">readout.py</a> reproduces the analysis in a fresh output with a600s alarm and blocked network/process/GPU access.</p></details>
 </body></html>'''
 (dest/'READOUT.html').write_text(text)
if __name__=='__main__':
 if len(sys.argv)!=2:raise SystemExit('readout.py FRESH_ABSOLUTE_OUTPUT_DIRECTORY')
 main(sys.argv[1])
