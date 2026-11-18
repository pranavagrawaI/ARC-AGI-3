"""Run run_peak.py only when the uplink can carry the 928 KB SaveKernel body inside Kaggle's ~10 s window.

On 2026-10-30 the hotspot uplink ran at 20-50 KB/s; SaveKernel returned 499 with an empty body after ~10 s on every
peak push, and a padded 928 KB CPU probe reproduced it (200 KB saved in 10.3 s). A failed save is archived and another
push allowed only after non-acceptance is verified: both slugs denied, no new own kernel, no active session, 0 reserved.
"""
import datetime as dt,json,shutil,subprocess,sys,time
from pathlib import Path
import run_pair as pair
import run_peak as peak

HERE=Path(__file__).resolve().parent
RUN=peak.RUN;FOLDER=RUN/peak.ARM
KPY=pair.KPY
MIN_UPLINK_BPS=130_000
DEADLINE=dt.datetime(2026,10,30,14,0,tzinfo=dt.timezone.utc)  # leaves ~85 min queue + ~75 min run before 17:30Z
MAX_PUSHES=5
PROBE=Path('/tmp/peak-uplink-probe.bin')

def log(**item):
 with (RUN/'push-gate.jsonl').open('a') as f:f.write(json.dumps({'utc':pair.now(),**item})+'\n')
 print(pair.now(),json.dumps(item),flush=True)

def uplink_bps():
 if not PROBE.exists():PROBE.write_bytes(__import__('os').urandom(150_000))
 r=subprocess.run(['curl','-s','-o','/dev/null','-w','%{speed_upload}','-X','POST','--data-binary','@'+str(PROBE),'https://httpbin.org/post','--max-time','30'],capture_output=True,text=True)
 try:return float(r.stdout)
 except ValueError:return 0.0

def verify_unaccepted():
 pair.RUN=Path('/tmp');api,lw=pair.api_setup();pair.RUN=RUN  # read-only checks; nothing written under the run
 lw.NOTEBOOK_DIR=HERE/'prepared'/peak.ARM
 used,q=lw.quota_used_s(api)
 out={'utc':pair.now(),'absent':{r:peak.slug_absent(api,r) for r in ('pranavagrawai/m2-animation-peak-screen1-20261027',peak.TITLE_SLUG)},
      'active':lw.active_sessions(api),'reserved':q['gpuQuotaSeconds']['timeReserved'],'used_plus_reserved_s':used,
      'newest_created':[k.ref for k in api.kernels_list(mine=True,user='pranavagrawai',sort_by='dateCreated',page_size=3)]}
 out['unaccepted']=all(out['absent'].values()) and not any(r and 'peak' in r for r,_ in out['active']) and out['reserved']==0 and not any('peak' in r for r in out['newest_created'])
 return out

def save_failed_with_499():
 attempt=FOLDER/'launch-attempt.json';http=FOLDER/'push-http.json'
 if not attempt.exists() or json.loads(attempt.read_text()).get('state')=='accepted' or not http.exists():return False
 saves=[e for e in json.loads(http.read_text()) if e['url'].endswith('/SaveKernel')]
 return bool(saves) and all(e['status']==499 and not e['body_head'] for e in saves)

def main():
 pushes=len(list(RUN.glob('attempt-*-unaccepted')))
 while dt.datetime.now(dt.timezone.utc)<DEADLINE:
  if (FOLDER/'launch-attempt.json').exists():
   if not save_failed_with_499():log(phase='STOP',reason='existing launch marker is not a verified-499 save failure');return
   check=verify_unaccepted()
   if not check['unaccepted']:log(phase='STOP',reason='non-acceptance not verified; never push',check=check);return
   pushes+=1;archive=RUN/f'attempt-{pushes}-unaccepted';archive.mkdir()
   for p in list(FOLDER.iterdir())+[RUN/'PEAK.json',RUN/'state.json']:
    if p.exists():shutil.move(str(p),archive/p.name)
   pair.save(archive/'RESOLUTION.json',{**check,'finding':'NOT_ACCEPTED','cause':'SaveKernel 499 empty body under slow uplink'})
   log(phase='ARCHIVED',attempt=pushes)
  if pushes>=MAX_PUSHES:log(phase='STOP',reason='push budget exhausted');return
  bps=uplink_bps();log(phase='UPLINK',bps=bps)
  if bps<MIN_UPLINK_BPS:time.sleep(120);continue
  log(phase='LAUNCH',attempt=pushes+1)
  rc=subprocess.run([KPY,'-B',str(HERE/'run_peak.py')],cwd=HERE).returncode
  if (FOLDER/'launch-response.json').exists():log(phase='DONE',rc=rc);return
 log(phase='STOP',reason='gate deadline reached without an accepted push')

if __name__=='__main__':main()
