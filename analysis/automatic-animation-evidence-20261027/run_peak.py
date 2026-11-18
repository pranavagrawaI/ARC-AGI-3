"""Treatment-only successor to the closed pair: one peak push, queue and runtime clocks observed separately.

The closed pair's observer started its 7,200 s deadline at submission, so ~85 min QUEUED exhausted it during a healthy
control. Here queue time has its own bound, the runtime clock starts at the first non-QUEUED observation, both clocks
are rebuilt from status-history.jsonl on restart, and any existing launch marker forbids a second push.
"""
import datetime as dt,fcntl,hashlib,json,os,subprocess,sys,time
from pathlib import Path
import run_pair as pair

HERE=Path(__file__).resolve().parent
RUN=HERE/'run-peak-20261030'
pair.RUN=RUN  # pair.state/preflight/source_readback/collect write under this run, never under the closed run-screen1
ARM='peak'
PREPARED=HERE/'prepared'/ARM  # kernel dir pushed; run_peak_slim.py points it at the slim kernel
QUEUE_CAP_S=5*3600
RUNTIME_CAP_S=6000+1800  # Kaggle enforces the 6000 s kernel timeout; margin covers status lag and export
MAX_STATUS_ERRORS=15
TERMINAL={'COMPLETE','ERROR','CANCELLED','CANCELED','CANCEL_ACKNOWLEDGED'}
CONTROL_HEALTH=HERE/'run-screen1/recovery-20261027/HEALTH.json'
NO_HARM_MARGIN=7
TITLE_SLUG='pranavagrawai/m2-automatic-animation-peak-screen1-20261027'  # Kaggle derives the slug from the title

def ts(utc):return dt.datetime.fromisoformat(utc).timestamp()

def history(path):
 path=Path(path)
 return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []

def clocks(entries,accepted_utc):
 """Queue clock from acceptance; runtime clock from the first observation that is neither QUEUED nor an error."""
 started=next((e['utc'] for e in entries if e.get('status') and e['status'].split('.')[-1]!='QUEUED'),None)
 return ts(accepted_utc),(ts(started) if started else None)

def observe(status_fn,accepted_utc,history_path,reauth=lambda:None,clock=time.time,sleep=time.sleep,poll_s=120):
 """Return the terminal status string; raise on a breached queue or runtime bound. Restart-safe via history_path."""
 errors=0
 while True:
  queued_at,running_at=clocks(history(history_path),accepted_utc)
  t=clock()
  if running_at is None and t>queued_at+QUEUE_CAP_S:raise RuntimeError('queue-observation bound exceeded; accepted arm unresolved, collect later, never re-push')
  if running_at is not None and t>running_at+RUNTIME_CAP_S:raise RuntimeError('runtime-observation bound exceeded; accepted arm unresolved, collect later, never re-push')
  try:
   status=str(status_fn());errors=0
   item={'utc':dt.datetime.fromtimestamp(t,dt.timezone.utc).isoformat(),'status':status}
  except Exception as error:
   errors+=1
   item={'utc':dt.datetime.fromtimestamp(t,dt.timezone.utc).isoformat(),'status':None,'error':type(error).__name__,'http_status':getattr(getattr(error,'response',None),'status_code',None)}
  with Path(history_path).open('a') as f:f.write(json.dumps(item)+'\n')
  if item['status'] and item['status'].split('.')[-1] in TERMINAL:return item['status']
  if errors>=MAX_STATUS_ERRORS:raise RuntimeError('status observation failed repeatedly; accepted arm unresolved, never re-push')
  if errors:reauth()
  sleep(poll_s)

def launch_state(folder):
 """None = no attempt ever made; 'accepted' = ref known; anything else = acceptance unresolved (stop, never push)."""
 marker=Path(folder)/'launch-attempt.json'
 if not marker.exists():return None
 return json.loads(marker.read_text()).get('state')

def no_harm(peak_levels,control_levels):
 return {'control_levels':control_levels,'peak_levels':peak_levels,'floor':control_levels-NO_HARM_MARGIN,'pass':peak_levels>=control_levels-NO_HARM_MARGIN}

def slug_absent(api,ref):
 try:api.kernels_status(ref)
 except Exception as error:
  # The SDK reports an unknown own slug as ValueError "Permission 'kernels.get' was denied", not an HTTP error.
  denied=isinstance(error,ValueError) and "Permission 'kernels.get' was denied" in str(error)
  return denied or getattr(getattr(error,'response',None),'status_code',None) in {403,404}
 return False

def push_once(folder):
 if (HERE/'run-screen1'/ARM).exists():raise RuntimeError('closed pair holds a peak attempt; successor refuses')
 control=json.loads(CONTROL_HEALTH.read_text())
 if not control['valid']:raise RuntimeError('recovered control is not healthy')
 api,pf=pair.preflight(ARM,6000)
 intended='pranavagrawai/'+json.loads((PREPARED/'kernel-metadata.json').read_text())['id'].split('/')[-1]
 absent={ref:slug_absent(api,ref) for ref in (intended,TITLE_SLUG)}
 pair.save(folder/'slug-absence.json',{'utc':pair.now(),'absent':absent})
 if not all(absent.values()):raise RuntimeError('a peak kernel slug already exists; a push would not be a fresh v1')
 receipt={'utc':pair.now(),'arm':ARM,'hard_cap_s':6000,'preflight':pf,'successor_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=pair.ROOT,text=True).strip()}
 def push():
  # Two pushes on 10-30 died in kagglesdk on an unparseable application/json body; record the raw HTTP exchange.
  import requests
  send=requests.Session.send;exchanges=[]
  def recording(session,request,**kwargs):
   r=send(session,request,**kwargs)
   exchanges.append({'method':request.method,'url':request.url.split('?')[0],'status':r.status_code,'headers':{k:v for k,v in r.headers.items() if k.lower() in {'content-type','content-length','date','server','x-cloud-trace-context','retry-after'}},'body_head':r.text[:2000]})
   return r
  requests.Session.send=recording
  try:response=json.loads(str(api.kernels_push(str(PREPARED),timeout='6000')))
  finally:
   requests.Session.send=send;pair.save(folder/'push-http.json',exchanges)
  if response.get('error') or response.get('versionNumber')!=1:raise RuntimeError('push not accepted as a fresh v1')
  return response
 pair.state('PUSHING',arm=ARM)
 pair.authorize_launch(folder,receipt,push)
 return api

def collect_bounded(ref):
 folder=RUN/ARM
 with (folder/'collector.stdout').open('x') as stdout,(folder/'collector.stderr').open('x') as stderr:
  result=subprocess.run([pair.KPY,'-B',str(Path(__file__).resolve()),'collect',ref],stdout=stdout,stderr=stderr,timeout=900)
 if result.returncode:raise RuntimeError('output collection failed; partial evidence preserved')

def main():
 RUN.mkdir(exist_ok=True);folder=RUN/ARM;folder.mkdir(exist_ok=True)
 with (RUN/'driver.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  with (RUN/'driver-invocations.jsonl').open('a') as f:f.write(json.dumps({'utc':pair.now(),'pid':os.getpid()})+'\n')
  try:
   pair.verify_sources()
   launched=launch_state(folder)
   if launched is None:push_once(folder)
   elif launched!='accepted':raise RuntimeError('launch acceptance unresolved ('+launched+'); refusing any push')
   response=json.loads((folder/'launch-response.json').read_text());ref=response['ref'].split('/code/')[-1]
   accepted_utc=json.loads((folder/'launch-attempt.json').read_text())['utc']
   pair.state('ACCEPTED',arm=ARM,ref=ref)
   api,lw=pair.api_setup()
   if not (folder/'source-readback').exists():pair.source_readback(api,ARM,ref)
   holder={'api':api}
   def reauth():
    lw.refresh_token();holder['api']=lw.KaggleApi();holder['api'].authenticate()
   def status_fn():
    status=holder['api'].kernels_status(ref).status;pair.state('OBSERVING',arm=ARM,ref=ref,status=str(status));return status
   status=observe(status_fn,accepted_utc,folder/'status-history.jsonl',reauth=reauth)
   pair.state('COLLECTING',arm=ARM,ref=ref,status=status)
   if not (folder/'output').exists():collect_bounded(ref)
   expected={g['game_id'] for g in json.loads(pair.EXPECTED.read_text())['game_runs']}
   gate=pair.assess_output(folder/'output',status,expected);pair.save(folder/'HEALTH.json',gate)
   if not gate['valid']:raise RuntimeError('unhealthy peak: '+', '.join(gate['issues']))
   stats=pair.request_receipts(folder/'output')
   if not stats['totals'].get('automatic_image_request_exposures',0):raise RuntimeError('peak flag ineffective: zero automatic-image exposures')
   _,lw=pair.api_setup();used,quota=lw.quota_used_s(api)
   pair.save(folder/'quota-terminal.json',{'utc':pair.now(),'quota':quota,'used_plus_reserved_s':used})
   control=json.loads(CONTROL_HEALTH.read_text())
   result={'utc':pair.now(),'status':'AWAITING_TRACE_REVIEW','peak':gate,'control':control,'no_harm':no_harm(gate['levels'],control['levels']),
           'score_delta':gate['mean_score']-control['mean_score'],'automatic_images':stats['totals'],'adoption':False}
   pair.save(RUN/'PEAK.json',result);pair.state('AWAITING_TRACE_REVIEW',result=result)
  except BaseException as error:
   safe=str(error)[:600] if not hasattr(error,'response') else 'HTTP status '+str(getattr(error.response,'status_code',None))
   result={'utc':pair.now(),'status':'STOPPED','error_type':type(error).__name__,'reason':safe,'retry_authorized':False,'adoption':False}
   pair.save(RUN/'PEAK.json',result);pair.state('STOPPED',result=result)
   raise SystemExit(1)

if __name__=='__main__':
 if sys.argv[1:2]==['collect']:pair.collect(ARM,sys.argv[2])
 else:main()
