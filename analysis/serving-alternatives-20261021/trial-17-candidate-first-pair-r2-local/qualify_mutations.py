"""Two isolated CPU regression witnesses; approved/prior sources never edited."""
import hashlib,json,os,shutil,subprocess,sys,tempfile
from pathlib import Path
HERE=Path(__file__).resolve().parent

def qualify(destination):
 destination=Path(destination);destination.mkdir(exist_ok=False);spec=json.loads((HERE/'packet-spec.json').read_bytes());rows=[]
 cases=[('missing_reverse_OOM_boundary','pair_runner.py',"self.boundary('candidate.released.before_fresh')","pass # deliberate isolated missing OOM boundary",'test_oom_finalization.py','OOM counter increased'),
  ('missing_actual_dispatch_guard','phase_tools.py',"if name=='control.prepare':self.control_gate(end)","if name=='control.prepare':pass # deliberate isolated missing guard",'test_actual_dispatch.py','not raised')]
 for name,filename,before,after,test,expected in cases:
  out=destination/name;out.mkdir()
  with tempfile.TemporaryDirectory() as td:
   root=Path(td)
   for member in set(spec['files'])|{'supervisor.py','http_once.py','packet-spec.json','candidate-contract.json'}:
    target=root/member;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(HERE/member,target)
   (root/'tests').mkdir()
   for member in ('test_oom_finalization.py','test_actual_dispatch.py','cpu_candidate.py'):shutil.copy2(HERE/'tests'/member,root/'tests'/member)
   source=(root/filename).read_text()
   if source.count(before)!=1:raise ValueError('exact mutation source not found')
   source=source.replace(before,after);(root/filename).write_text(source)
   env=dict(os.environ,PACKET_PROOF_DIR=str((out/'artifacts').resolve()))
   command=[sys.executable,'-m','unittest','discover','-s','tests','-p',test]
   completed=subprocess.run(command,cwd=root,capture_output=True,text=True,env=env,timeout=30)
   (out/'stdout').write_text(completed.stdout);(out/'stderr').write_text(completed.stderr)
   detected=completed.returncode!=0 and 'AssertionError' in completed.stderr and expected in completed.stderr
   row=dict(case=name,changed_file=filename,variant_sha256=hashlib.sha256(source.encode()).hexdigest(),command=command,exit_code=completed.returncode,expected_assertion=expected,detected=detected,network_or_GPU_authority=False)
   (out/'RESULT.json').write_text(json.dumps(row,indent=2)+'\n');rows.append(row)
   if not detected:raise ValueError('isolated CPU regression did not detect '+name)
 return rows
