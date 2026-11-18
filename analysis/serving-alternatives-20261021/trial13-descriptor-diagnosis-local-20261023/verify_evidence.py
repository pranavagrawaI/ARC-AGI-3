"""Check the preserved CPU experiment against independent declared expectations."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
count=0

def check(ok,why):
 global count
 if not ok:raise ValueError(why)
 count+=1

for name,h in json.loads((ROOT/'SOURCE-BINDINGS.json').read_text())['references'].items():check(hashlib.sha256(Path(name).read_bytes()).hexdigest()==h,'original input drift '+name)
for directory in ('qualification-r1','qualification-python312'):
 p=ROOT/directory
 def result(v,c):return json.loads((p/(v+'-'+c)/'RESULT.json').read_text())
 low=result('baseline','direct_low');high=result('baseline','direct_high');write=result('baseline','write_high');pollhigh=result('poll','direct_high');pollwrite=result('poll','write_high')
 check(low['operation_completed'],'original low-fd frame failed')
 check(high['error_type']=='ValueError' and 'read_exact' in high['traceback'] and high['read_fd']>=2048,'original high-fd read witness lost')
 check(write['error_type']=='ValueError' and write['write_fd']>=2048,'original high-fd blocked-write witness lost')
 check(pollhigh['operation_completed'],'poll high-fd read failed')
 check(pollwrite['error_type']=='TransferDeadline' and pollwrite['finished']>=pollwrite['absolute_end'],'poll timeout/deadline weakened')
 for variant in ('baseline','lifecycle','poll'):
  r=result(variant,'pressure');check(r['health_files']==524,'wrong controlled poll count')
  check(all(v['pidfd_exited'] for v in r['retained_fixture_cleanup']),'fixture process not observed exited')
  check(r['live_http_log_handles_open'],'active server handles closed too early')
  if variant=='baseline':check(r['error_type']=='ValueError' and 'read_exact' in r['traceback'] and r['retained_parent_handles']==1050,'retention failure witness lost')
  else:check(r['operation_completed'] and set(r['observation']['probes'])=={'gpu','apps','cgroup','port','disk','native'} and not r['observation']['probe_worker_alive'],'complete bounded CPU observation missing')
  check(r['retained_parent_handles']==(2 if variant=='lifecycle' else 1050),'alternative handle behavior mismatch')
 r=result('baseline','pressure');trace=p/'baseline-pressure/phase/supervisor.jsonl';rows=[json.loads(s) for s in trace.read_text().splitlines()];check(any(e['event']=='ready' for e in rows),'actual HTTP200 path not reached')
selected=json.loads((ROOT/'selected-pressure-python312/RESULT.json').read_text());check(selected['operation_completed'] and selected['health_files']==524 and selected['retained_parent_handles']==2,'selected pressure failed')
check(all(v['pidfd_exited'] for v in selected['retained_fixture_cleanup']),'selected fixture cleanup failed')
for name in ('framing-baseline','framing-lifecycle','framing-poll','framing-poll-high','selected-contract-proof'):
 r=json.loads((ROOT/name/'RESULT.json').read_text());check(r['pass_'] and r['tests']==7,'contract suite failed '+name)
# Selected source changes only phase lifetime handling; IPC/identity/filter remain exact.
for file in (ROOT/'baseline').iterdir():
 if file.is_file() and file.name!='phase_tools.py':check(file.read_bytes()==(ROOT/'selected'/file.name).read_bytes(),'unrelated selected source change '+file.name)
print(json.dumps(dict(status='SOURCE_BOUND_CPU_DIAGNOSIS_ONLY',checks=count,selected_phase_tools_sha256=hashlib.sha256((ROOT/'selected/phase_tools.py').read_bytes()).hexdigest(),selected_probe_ipc_unchanged=True,remote_failure_attribution='UNPROVEN_NO_REMOTE_FD_OR_STACK',cleanup_identity_error='SEPARATE_UNEXPLAINED',no_gpu_packet=True)))
