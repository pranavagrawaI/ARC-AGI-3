"""Continue the authorized pair after a pre-push slot hold; never retry a GPU attempt."""
import datetime as dt,fcntl,hashlib,json,shutil,time
from pathlib import Path
import run_pair as pair

def no_gpu_attempts():
 attempts=list(pair.RUN.glob('*/launch-attempt.json'))+list(pair.RUN.glob('*/launch-response.json'))
 if attempts:raise RuntimeError('a GPU attempt already exists; admission continuation forbidden')

def main():
 no_gpu_attempts();binding=pair.verify_sources()
 archive=pair.RUN/'pre-push-slot-hold';archive.mkdir(exist_ok=False)
 for rel in ['driver-started.json','PAIR.json','state.json','control/preflight.json']:
  src=pair.RUN/rel
  if src.exists():shutil.copy2(src,archive/rel.replace('/','-'))
 receipt={'utc':pair.now(),'reason':'original driver refused admission before any push because keepalive test occupies slot','frozen_pair_sha256':binding,'original_gpu_attempts':0,'wait_cap_s':10800,'pair_gpu_cap_s':12000,'watch_ref':'pranavagrawai/m2-f1x-keepalive','wrapper_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 with (pair.RUN/'admission-continuation.json').open('x') as f:json.dump(receipt,f,indent=2)
 # Preserve original preflight failure as a separate local invocation.
 for name in ['PAIR.json','state.json']:
  src=pair.RUN/name
  if src.exists():src.rename(archive/('original-'+name))
 api,lw=pair.api_setup();deadline=time.time()+10800
 watch=receipt['watch_ref']
 while time.time()<deadline:
  lw.refresh_token();api=lw.KaggleApi();api.authenticate()
  status=str(api.kernels_status(watch).status)
  with (pair.RUN/'admission-history.jsonl').open('a') as f:f.write(json.dumps({'utc':pair.now(),'ref':watch,'status':status})+'\n')
  pair.state('WAITING_FOR_SLOT',watch_ref=watch,watch_status=status,deadline_utc=dt.datetime.fromtimestamp(deadline,dt.timezone.utc).isoformat())
  if status.split('.')[-1] in {'COMPLETE','ERROR','CANCELLED','CANCELED'}:
   lw.NOTEBOOK_DIR=pair.HERE/'prepared/control'
   active=lw.active_sessions(api);used,q=lw.quota_used_s(api)
   if pair.admit(active,used,q['gpuQuotaSeconds']['timeReserved'],12000):
    no_gpu_attempts();pair.verify_sources()
    (pair.RUN/'driver-started.json').rename(archive/'moved-driver-started.json')
    pair.main();return
   if active and active[0][0]:watch=active[0][0]
  time.sleep(min(120,max(0,deadline-time.time())))
 pair.state('ADMISSION_EXPIRED',reason='no GPU launch; three-hour slot-wait cap expired')
if __name__=='__main__':
 try:main()
 except BaseException as error:
  if isinstance(error,SystemExit):raise
  pair.state('ADMISSION_ERROR',error_type=type(error).__name__,reason=str(error)[:400])
  raise SystemExit(1)
