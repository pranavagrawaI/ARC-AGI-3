"""Bounded CPU/source-path comparison, no packet/launch authority."""
import argparse,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(exist_ok=False);signal.alarm(600);start=time.monotonic()
refs=json.loads((ROOT/'SOURCE-BINDINGS.json').read_text())['references']
def verify():
 for name,h in refs.items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h,name
verify();cases=[]
for variant,case in [('baseline','direct_low'),('baseline','direct_high'),('poll','direct_high'),('baseline','write_high'),('poll','write_high'),('baseline','pressure'),('lifecycle','pressure'),('poll','pressure')]:
 target=out/(variant+'-'+case)
 env=dict(os.environ,LOCAL_ADAPTER_PROOF='1',PYTHONPATH=str(ROOT.parent/'trial-13-post-control-load-r2-local/guard'))
 r=subprocess.run([sys.executable,str(ROOT/'run_case.py'),variant,case,str(target)],env=env,capture_output=True,text=True,timeout=max(1,590-(time.monotonic()-start)))
 target.mkdir(exist_ok=True);(target/'stdout').write_text(r.stdout);(target/'stderr').write_text(r.stderr);assert r.returncode==0,r.stderr
 result=json.loads((target/'RESULT.json').read_text());cases.append(result)
 print(r.stdout.strip(),flush=True)
verify();summary=dict(status='CPU_SOURCE_PATH_DIAGNOSIS_NOT_REMOTE_ATTRIBUTION',seconds=time.monotonic()-start,cases=[{k:c.get(k) for k in ('variant','case','operation_completed','error_type','after_ready_fds','retained_parent_handles','health_files','select_calls')} for c in cases],bound_references=len(refs),gpu_work=False,external_network=0,remote_fd_and_stack_unavailable=True,cleanup_identity_error_not_explained=True)
(out/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n');signal.alarm(0)
