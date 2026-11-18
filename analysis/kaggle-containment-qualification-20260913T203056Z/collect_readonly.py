"""Post-admission read-only exact returned identity; no retry or mutation."""
from pathlib import Path
import datetime,hashlib,json,re,shutil,subprocess,tempfile,time
HERE=Path(__file__).resolve().parent
CLI='/home/pranav/.local/bin/kaggle'
def call(args,name):
 p=subprocess.run([CLI,*args],stdin=subprocess.DEVNULL,close_fds=True,capture_output=True,text=True,timeout=90)
 r={'arguments':args,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 (HERE/name).write_text(json.dumps(r,indent=2)+'\n');return r

def main():
 launch=json.loads((HERE/'launch.json').read_text());assert launch['returncode']==0
 match=re.search(r'Kernel version (\d+) successfully pushed\.  Please check progress at https://www.kaggle.com/code/(\S+)',launch['stdout']);assert match
 kernel=match[2];version=int(match[1]);assert version==1;exact=kernel+'/'+str(version)
 while True:
  r=call(['kernels','status',exact],'status-exact-version-final.json')
  m=re.search(r'KernelWorkerStatus\.([A-Z_]+)',r['stdout']);r['status']=m[1] if m and r['returncode']==0 else 'UNKNOWN'
  r['since_launch_s']=time.time()-json.loads((HERE/'LAUNCH-ATTEMPT.json').read_text())['epoch']
  (HERE/'status-exact-version-final.json').write_text(json.dumps(r,indent=2)+'\n')
  with (HERE/'monitor-returned.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
  if r['status'] in ('COMPLETE','ERROR','CANCELLED','CANCELED'):break
  if r['since_launch_s']>600:raise RuntimeError('REMOTE_NOT_TERMINAL')
  time.sleep(10)
 call(['kernels','files',exact],'files-exact-version.json')
 retained=[];discarded=[]
 allow=r'^(qualification/(result|platform|(candidate|inspection)-(benign|filesystem|inherited|syscalls|readonly|wall|output|forged)-rootfs)[.]json|package-(lifecycle|worker)[.]json)$'
 with tempfile.TemporaryDirectory(prefix='containment-retrieval-') as temp:
  rr=call(['kernels','output',exact,'--path',temp,'--force','--file-pattern',allow],'output-retrieval.json');assert rr['returncode']==0
  for p in Path(temp).rglob('*'):
   if not p.is_file():continue
   rel=p.relative_to(temp).as_posix()
   if re.fullmatch(allow,rel) and not p.is_symlink():
    dest=HERE/'remote'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    retained.append({'path':rel,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()})
   else:discarded.append(rel);p.unlink()
 (HERE/'RETRIEVAL.json').write_text(json.dumps({'kernel':kernel,'version':version,'retained':retained,'discarded_unread':discarded},indent=2)+'\n')
 call(['quota','--format','json'],'quota-after.json')
 print(json.dumps({'terminal':r,'retained':retained},indent=2))
if __name__=='__main__':main()
