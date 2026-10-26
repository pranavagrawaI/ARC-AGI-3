"""One authorized serial pair; durable launch markers prohibit duplicate pushes."""
import datetime as dt,json,re
from pathlib import Path
def save(path,value):
 path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def authorize_launch(folder,receipt,push):
 folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
 marker=folder/'launch-attempt.json'
 with marker.open('x') as f:json.dump({**receipt,'state':'attempting'},f,indent=2)
 try:
  result=push()
 except BaseException as error:
  save(marker,{**receipt,'state':'acceptance unresolved; never retry','error':repr(error)[:500]})
  raise
 save(folder/'launch-response.json',result)
 save(marker,{**receipt,'state':'accepted','response':result})
 return result
def admit(active,used_plus_reserved,reserved,remaining_cap):
 return not active and reserved==0 and used_plus_reserved>=0 and used_plus_reserved+remaining_cap<=108000
def assess_output(folder,status,expected_ids):
 folder=Path(folder);issues=[]
 if status.split('.')[-1]!='COMPLETE':issues.append('platform not COMPLETE')
 required=[folder/'benchmark.json',folder/'serve.log']
 if any(not p.is_file() or not p.stat().st_size for p in required):return {'valid':False,'issues':issues+['missing benchmark or serve log']}
 b=json.loads(required[0].read_text());games=b['game_runs']
 if len(games)!=25 or {g['game_id'] for g in games}!=set(expected_ids):issues.append('game identity/count mismatch')
 logs='\n'.join(p.read_text(errors='replace') for p in folder.glob('*.log') if p.name!='serve.log')
 if 'GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition capability: (12, 0)' not in logs:issues.append('actual RTX PRO 6000 identity not demonstrated')
 ready=re.findall(r'SERVER READY at (\d+)s',logs)
 if not ready or int(ready[-1])>1300:issues.append('readiness missing or late')
 duration=(dt.datetime.fromisoformat(b['end_time'])-dt.datetime.fromisoformat(b['start_time'])).total_seconds()
 if not 3600<=duration<=3900:issues.append('one-hour gameplay horizon not demonstrated')
 if any(g['state'] not in {'won','cancelled'} for g in games):issues.append('unexpected game state')
 if any(g['solver_note'] and not g['solver_note'].startswith('tokens=') for g in games):issues.append('game error-ending note')
 if any(not g['history'] for g in games):issues.append('game had no recorded action')
 if any(not (folder/(g['game_id']+'_p0_requests.jsonl')).is_file() for g in games):issues.append('missing request trace')
 serve=required[1].read_text(errors='replace')
 if re.search(r'OutOfMemoryError|CUDA out of memory|CUDA error:|Watchdog timeout',serve):issues.append('fatal serving error')
 return {'valid':not issues,'issues':issues,'duration_s':duration,'ready_s':int(ready[-1]) if ready else None,'levels':sum(g['levels_completed'] for g in games),'mean_score':sum(g['final_score'] for g in games)/25,'wins':sum(g['state']=='won' for g in games)}

# Runtime orchestration uses the existing field-tested Kaggle helpers unchanged.
import fcntl,hashlib,os,subprocess,sys,time
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
RUN=HERE/'run-screen1'
KPY='/home/pranav/.local/share/uv/tools/kaggle/bin/python'
EXPECTED=ROOT/'solvers/m2_f1x_reap/experiment/screen-1/merge/output/benchmark.json'

def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def state(phase,**detail):
 save(RUN/'state.json',{'utc':now(),'pid':os.getpid(),'phase':phase,**detail})
 print(now(),phase,json.dumps(detail),flush=True)
def verify_sources():
 bindings=json.loads((HERE/'FROZEN.json').read_text())
 for rel,sha in bindings['sha256'].items():
  if digest(ROOT/rel)!=sha:raise ValueError('frozen source drift: '+rel)
 return digest(HERE/'FROZEN.json')
def api_setup():
 import requests
 original=requests.Session.send
 def bounded(session,request,**kwargs):
  kwargs.setdefault('timeout',(15,45));return original(session,request,**kwargs)
 requests.Session.send=bounded
 sys.path.insert(0,str(ROOT/'analysis/mtp-depth-20261006'))
 import launch_watch as lw
 lw.EVID=RUN;lw.refresh_token();api=lw.KaggleApi();api.authenticate()
 return api,lw

def preflight(arm,remaining_cap):
 binding=verify_sources();api,lw=api_setup()
 lw.NOTEBOOK_DIR=HERE/'prepared'/arm
 active=lw.active_sessions(api);used,quota=lw.quota_used_s(api)
 reserved=quota['gpuQuotaSeconds']['timeReserved']
 result={'utc':now(),'active':active,'quota':quota,'used_plus_reserved_s':used,'remaining_cap_s':remaining_cap,'binding_sha256':binding,'admitted':admit(active,used,reserved,remaining_cap)}
 save(RUN/arm/'preflight.json',result)
 if not result['admitted']:raise RuntimeError('preflight refused exclusive slot or quota')
 return api,result

def source_readback(api,arm,ref):
 folder=RUN/arm/'source-readback';folder.mkdir(exist_ok=False)
 api.kernels_pull(ref,str(folder),metadata=True,quiet=True)
 paths=list(folder.glob('*.ipynb'))
 if len(paths)!=1:raise RuntimeError('source readback missing or ambiguous')
 planned=json.loads(next((HERE/'prepared'/arm).glob('*.ipynb')).read_text())
 remote=json.loads(paths[0].read_text())
 project=lambda nb:[(c['cell_type'],''.join(c['source'])) for c in nb['cells']]
 exact=project(remote)==project(planned)
 save(folder/'identity.json',{'utc':now(),'ref':ref,'version':1,'source_equal':exact,'readback_sha256':digest(paths[0])})
 if not exact:raise RuntimeError('remote notebook source differs')

def wait_terminal(api,arm,ref,started_utc):
 deadline=dt.datetime.fromisoformat(started_utc).timestamp()+7200
 while time.time()<deadline:
  observation=api.kernels_status(ref);status=str(observation.status)
  item={'utc':now(),'status':status,'failure':getattr(observation,'failure_message',None)}
  with (RUN/arm/'status-history.jsonl').open('a') as f:f.write(json.dumps(item)+'\n')
  state('RUNNING',arm=arm,ref=ref,status=status)
  if status.split('.')[-1] in {'COMPLETE','ERROR','CANCELLED','CANCELED'}:return status
  time.sleep(min(120,max(0,deadline-time.time())))
 raise RuntimeError('terminal-observation wall deadline exceeded; no next arm')

def collect(arm,ref):
 api,_=api_setup()
 # The SDK downloads all pages but does not raise on HTTP error bodies by default.
 import requests
 get=requests.get
 def checked_get(*args,**kwargs):
  response=get(*args,**kwargs);response.raise_for_status();return response
 requests.get=checked_get
 folder=RUN/arm;part=folder/'output.partial';part.mkdir(exist_ok=False)
 files,token=api.kernels_output(ref,path=str(part),force=False,quiet=True,page_size=100)
 if token:raise RuntimeError('incomplete output pagination')
 inventory={str(p.relative_to(part)):{'bytes':p.stat().st_size,'sha256':digest(p)} for p in part.rglob('*') if p.is_file()}
 if not inventory:raise RuntimeError('empty output collection')
 save(folder/'collection.json',{'utc':now(),'ref':ref,'files':inventory,'api_file_count':len(files),'complete_download':True})
 part.rename(folder/'output')

def collect_bounded(arm,ref):
 folder=RUN/arm
 with (folder/'collector.stdout').open('x') as stdout,(folder/'collector.stderr').open('x') as stderr:
  result=subprocess.run([KPY,'-B',str(Path(__file__).resolve()),'collect',arm,ref],stdout=stdout,stderr=stderr,timeout=900)
 if result.returncode:raise RuntimeError('output collection failed; partial evidence preserved')

def request_receipts(folder):
 import base64,collections
 totals=collections.Counter();per_game={};images=[]
 image_dir=Path(folder).parent/'automatic-images';image_dir.mkdir(exist_ok=True)
 for p in sorted(Path(folder).glob('*_requests.jsonl')):
  local=collections.Counter();seen=set()
  with p.open() as stream:
   for line_no,line in enumerate(stream,1):
    r=json.loads(line)
    if r.get('event')=='response':
     local['responses']+=1;local['completion_tokens']+=int((r.get('usage') or {}).get('completion_tokens') or 0)
    if r.get('event')!='request':continue
    local['requests']+=1
    users=[m for m in r.get('messages',[]) if m.get('role')=='user']
    content=users[-1]['content'] if users else ''
    parts=[{'type':'text','text':content}] if isinstance(content,str) else content
    text='\n'.join(x.get('text','') for x in parts if isinstance(x,dict) and x.get('type')=='text')
    if 'left the board area unchanged, but it was not a no-op:' in text:local['qualifying_request_exposures']+=1
    for i,x in enumerate(parts):
     label=x.get('text','') if isinstance(x,dict) else ''
     if not label.startswith('Animation image (frame '):continue
     local['automatic_image_request_exposures']+=1
     if i+1>=len(parts) or parts[i+1].get('type')!='image_url':raise ValueError('automatic-image label lacks image')
     data=parts[i+1]['image_url']['url']
     if not data.startswith('data:image/png;base64,'):raise ValueError('unexpected automatic-image representation')
     raw=base64.b64decode(data.split(',',1)[1],validate=True);sha=hashlib.sha256(raw).hexdigest()
     key=(text,sha)
     if key in seen:continue
     seen.add(key);local['distinct_automatic_messages']+=1
     target=image_dir/(sha+'.png')
     if not target.exists():target.write_bytes(raw)
     images.append({'game':p.name.split('_p0_')[0],'request_file':p.name,'line':line_no,'analysis_step':r.get('analysis_step'),'action':r.get('action'),'label':label,'png_sha256':sha,'png':str(target.relative_to(Path(folder).parent))})
  if not local['responses']:raise ValueError('game trace has no model response: '+p.name)
  per_game[p.name]=dict(local);totals.update(local)
 result={'totals':dict(totals),'per_game':per_game,'images':images}
 save(Path(folder).parent/'request-receipts.json',result);return result

def run_arm(arm,remaining_cap):
 folder=RUN/arm;folder.mkdir(exist_ok=True)
 api,pf=preflight(arm,remaining_cap)
 receipt={'utc':now(),'arm':arm,'hard_cap_s':6000,'preflight':pf,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
 def push():
  response=json.loads(str(api.kernels_push(str(HERE/'prepared'/arm),timeout='6000')))
  if response.get('error') or response.get('versionNumber')!=1:raise RuntimeError('push not accepted as a fresh v1')
  return response
 state('PUSHING',arm=arm)
 response=authorize_launch(folder,receipt,push)
 ref=response['ref'].split('/code/')[-1]
 state('ACCEPTED',arm=arm,ref=ref,response=response)
 source_readback(api,arm,ref)
 status=wait_terminal(api,arm,ref,receipt['utc'])
 state('COLLECTING',arm=arm,ref=ref,status=status)
 collect_bounded(arm,ref)
 expected={g['game_id'] for g in json.loads(EXPECTED.read_text())['game_runs']}
 gate=assess_output(folder/'output',status,expected)
 save(folder/'HEALTH.json',gate)
 if not gate['valid']:raise RuntimeError('unhealthy '+arm+': '+', '.join(gate['issues']))
 stats=request_receipts(folder/'output')
 if arm=='control' and stats['totals'].get('automatic_image_request_exposures',0):raise RuntimeError('control unexpectedly exposed automatic images')
 _,lw=api_setup();used,quota=lw.quota_used_s(api)
 save(folder/'quota-terminal.json',{'utc':now(),'quota':quota,'used_plus_reserved_s':used})
 return gate

def main():
 RUN.mkdir(exist_ok=True)
 with (RUN/'driver.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  with (RUN/'driver-started.json').open('x') as f:json.dump({'utc':now(),'pid':os.getpid(),'authority':'User: Do it. Do not wait for me, just do it.'},f,indent=2)
  try:
   control=run_arm('control',12000)
   peak=run_arm('peak',6000)
   result={'utc':now(),'status':'AWAITING_TRACE_REVIEW','control':control,'peak':peak,'level_delta':peak['levels']-control['levels'],'score_delta':peak['mean_score']-control['mean_score'],'no_harm_floor':control['levels']-7,'no_harm_pass':peak['levels']>=control['levels']-7,'adoption':False}
   save(RUN/'PAIR.json',result);state('AWAITING_TRACE_REVIEW',result=result)
  except BaseException as error:
   safe=str(error)[:600] if not hasattr(error,'response') else 'HTTP status '+str(getattr(error.response,'status_code',None))
   result={'utc':now(),'status':'STOPPED','error_type':type(error).__name__,'reason':safe,'retry_authorized':False,'adoption':False}
   save(RUN/'PAIR.json',result);state('STOPPED',result=result)
   raise SystemExit(1)
if __name__=='__main__':
 try:
  if sys.argv[1:] and sys.argv[1]=='collect':collect(sys.argv[2],sys.argv[3])
  else:main()
 except Exception as error:
  print(json.dumps({'error_type':type(error).__name__,'http_status':getattr(getattr(error,'response',None),'status_code',None)}),flush=True)
  raise SystemExit(1)
