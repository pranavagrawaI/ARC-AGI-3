"""Peak screen over the slim transport: run_peak's one-push observer/collector, aimed at the 42 KB slim kernel.

The frozen peak notebook (2e4e661e...) cannot cross a 13-50 KB/s uplink inside SaveKernel's ~10 s window (HTTP 499 x3 on
10-30). prepare_slim.py moved cells 3, 6 and 15 into the private dataset m2-animation-peak-cell-bodies-20261031; the stubs
hash-check each body and hand it to the live IPython shell, so the executed cells equal the frozen ones (SLIM.json,
test_slim_ipython.py). Runs serially after D22, which owns today's GPU slot.

  run_peak_slim.py           wait for D22 to release the slot, then one push, observe, collect, gate
  run_peak_slim.py collect REF
"""
import datetime as dt,hashlib,json,subprocess,sys,time
from pathlib import Path
import run_pair as pair
import run_peak as peak

HERE=Path(__file__).resolve().parent
SLIM=HERE/'prepared/peak-slim-20261031'
KERNEL=SLIM/'kernel'
KERNEL_ID='pranavagrawai/m2-animation-peak-slim-20261031'
RUN=HERE/'run-peak-20261031'
D22=pair.ROOT/'analysis/f1x-reap14-keepalive-20261031'
D22_PUSH_CUTOFF=dt.datetime(2026,10,31,12,0,tzinfo=dt.timezone.utc)
FROZEN_PEAK_SHA='2e4e661e6b78843733d02157bcc95d170d51027f6aec90d5c04c914e7bcf3f5a'
WAIT_UNTIL=dt.datetime(2026,10,31,14,30,tzinfo=dt.timezone.utc)  # leaves ~85 min queue + ~75 min run before 17:30Z

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def verify_slim():
 s=json.loads((SLIM/'SLIM.json').read_text());issues=[]
 if s['source_sha256']!=FROZEN_PEAK_SHA or digest(HERE/'prepared/peak/m2-animation-peak-screen1-20261027.ipynb')!=FROZEN_PEAK_SHA:issues.append('frozen peak source drift')
 if digest(KERNEL/(KERNEL_ID.split('/')[-1]+'.ipynb'))!=s['slim_notebook_sha256']:issues.append('slim notebook drift')
 if digest(KERNEL/'kernel-metadata.json')!=s['metadata_sha256']:issues.append('slim metadata drift')
 if not s['reconstruction_equal']:issues.append('reconstruction not equal')
 for c in s['cells']:
  if digest(SLIM/'dataset'/c['file'])!=c['sha256']:issues.append('dataset body drift '+c['file'])
 trip=json.loads((SLIM/'dataset-roundtrip.json').read_text())
 if not trip['roundtrip_sha256_equal'] or not all(trip['roundtrip_sha256_equal'].values()):issues.append('remote dataset round-trip not verified')
 if issues:raise RuntimeError('slim binding: '+', '.join(issues))
 return digest(SLIM/'SLIM.json')

def d22_released(now=None,folder=D22):
 """D22 owns today's slot: proceed only after its smoke is terminal, it explicitly released, or its push window closed unused."""
 now=now or dt.datetime.now(dt.timezone.utc);folder=Path(folder)
 if (folder/'terminal.json').exists():return 'D22 smoke terminal'
 if (folder/'evidence/gpu-release-to-peak.json').exists():return 'D22 explicit release'
 if now>=D22_PUSH_CUTOFF and not (folder/'launch-attempt.json').exists():return 'D22 push window closed with no attempt'
 return None

def source_readback(api,arm,ref):
 folder=RUN/arm/'source-readback';folder.mkdir(exist_ok=False)
 api.kernels_pull(ref,str(folder),metadata=True,quiet=True)
 paths=list(folder.glob('*.ipynb'))
 if len(paths)!=1:raise RuntimeError('source readback missing or ambiguous')
 project=lambda nb:[(c['cell_type'],''.join(c['source']) if isinstance(c['source'],list) else c['source']) for c in nb['cells']]
 exact=project(json.loads(paths[0].read_text()))==project(json.loads(next(KERNEL.glob('*.ipynb')).read_text()))
 pair.save(folder/'identity.json',{'utc':pair.now(),'ref':ref,'version':1,'source_equal':exact,'readback_sha256':digest(paths[0])})
 if not exact:raise RuntimeError('remote notebook source differs from slim kernel')

def configure():
 peak.RUN=pair.RUN=RUN
 peak.TITLE_SLUG=KERNEL_ID
 pair.source_readback=source_readback
 base_push=peak.push_once
 def push_once(folder):
  verify_slim()
  reason=d22_released()
  if not reason:raise RuntimeError('D22 still owns the GPU slot; no push')
  pair.save(folder/'slot-handoff.json',{'utc':pair.now(),'reason':reason,'slim_sha256':digest(SLIM/'SLIM.json')})
  return base_push(folder)
 peak.push_once=push_once
 peak.collect_bounded=collect_bounded
 peak.PREPARED=KERNEL

def collect_bounded(ref):
 folder=RUN/peak.ARM
 with (folder/'collector.stdout').open('x') as stdout,(folder/'collector.stderr').open('x') as stderr:
  result=subprocess.run([pair.KPY,'-B',str(Path(__file__).resolve()),'collect',ref],stdout=stdout,stderr=stderr,timeout=900)
 if result.returncode:raise RuntimeError('output collection failed; partial evidence preserved')

def wait_for_slot():
 RUN.mkdir(exist_ok=True)
 while dt.datetime.now(dt.timezone.utc)<WAIT_UNTIL:
  reason=d22_released()
  with (RUN/'slot-wait.jsonl').open('a') as f:f.write(json.dumps({'utc':pair.now(),'released':reason})+'\n')
  if reason:return reason
  time.sleep(300)
 return None

if __name__=='__main__':
 configure()
 if sys.argv[1:2]==['collect']:pair.collect(peak.ARM,sys.argv[2])
 elif (RUN/peak.ARM/'launch-attempt.json').exists() or wait_for_slot():peak.main()
 else:
  RUN.mkdir(exist_ok=True);pair.save(RUN/'PEAK.json',{'utc':pair.now(),'status':'NOT_LAUNCHED','reason':'D22 held the slot past the wait cutoff','adoption':False})
