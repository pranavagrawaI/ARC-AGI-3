"""Live-capable once-only pair. The sole credential holder is this provider parent."""
import argparse,json,os,selectors,signal,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE/'engine')]
from common import TASK,CAPS,read,put,journal,encoded,digest,hash_file
from transport import Transport,Stop,ENDPOINT,initial_ledger,write

def verify_inventory():
 inv=read(HERE/'INVENTORY.json')
 for name,sha in inv['files'].items():
  p=HERE/name
  if p.is_symlink() or not p.is_file() or hash_file(p)!=sha:raise Stop('candidate drift: '+name)
 for path,sha in inv['runtime'].items():
  if hash_file(path)!=sha:raise Stop('runtime drift')
 return hash_file(HERE/'INVENTORY.json')

def gates(review,release,approval):
 sha=verify_inventory()
 for path,role,decision in [(review,'independent-reviewer','PASS'),(release,'Hermes','RELEASE'),(approval,'PM','APPROVED')]:
  if not path.is_absolute():raise Stop('absolute gate paths required')
  d=read(path)
  if any(d.get(k)!=v for k,v in {'task':TASK,'role':role,'decision':decision,'inventory_sha256':sha,'endpoint':ENDPOINT}.items()):raise Stop('missing/mismatched independent gate')
 if read(release).get('review_sha256')!=hash_file(review):raise Stop('release review binding')
 if read(approval).get('credential_use')!='sole-key-existing-loader':raise Stop('credential approval')
 return sha

def sandbox_command(mode,root,prefix,freeze,limit):
 mapped=os.geteuid()==0 and any(int(l.split()[2])>65534 for l in Path('/proc/self/uid_map').read_text().splitlines())
 unshare=['unshare']+([] if mapped else ['--user','--map-auto','--map-root-user'])+['--net']
 return unshare+[sys.executable,'-B',str(HERE/'worker.py'),mode,str(root/'child'),str(prefix),str(freeze),str(limit)]

def supervise(mode,root,prefix,freeze,transport=None,hard=None,limit=100,command=None):
 root.mkdir(exist_ok=False);started=time.monotonic();hard=hard or (CAPS[mode]['hard'] if mode in CAPS else 30)
 command=command or sandbox_command(mode,root,prefix,freeze,limit)
 put(root/'COMMAND.json',{'argv':command,'environment':{'PATH':'/usr/bin:/bin','LANG':'C.UTF-8','PYTHONDONTWRITEBYTECODE':'1','HOME':str(root),'UTILITY_PARENT_PID':str(os.getpid())},'hard_seconds':hard})
 env={'PATH':'/usr/bin:/bin','LANG':'C.UTF-8','PYTHONDONTWRITEBYTECODE':'1','HOME':str(root),'UTILITY_PARENT_PID':str(os.getpid())}
 failure=None;timed_out=False;buffer=b'';request_no=0
 with (root/'worker.stderr').open('xb') as err,(root/'worker.stdout.raw').open('xb') as rawout:
  proc=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,env=env,start_new_session=True,cwd=HERE)
  sel=selectors.DefaultSelector();sel.register(proc.stdout,selectors.EVENT_READ)
  try:
   while True:
    remaining=hard-(time.monotonic()-started)
    if remaining<=0:timed_out=True;raise TimeoutError('worker hard deadline')
    events=sel.select(min(.2,remaining))
    if not events:
     if proc.poll() is not None:break
     continue
    chunk=os.read(proc.stdout.fileno(),65536)
    if not chunk:break
    rawout.write(chunk);rawout.flush();buffer+=chunk
    if len(buffer)>8_000_001:raise ValueError('worker output bound')
    while b'\n' in buffer:
     line,buffer=buffer.split(b'\n',1)
     msg=json.loads(line)
     if not isinstance(msg,dict) or set(msg)!={'request'} or transport is None:raise ValueError('worker protocol')
     request_no+=1
     reply=transport(msg['request'])
     if time.monotonic()-started>=hard:timed_out=True;raise TimeoutError('late worker reply')
     proc.stdin.write(encoded({'response':reply}));proc.stdin.flush()
   proc.wait(timeout=max(.001,hard-(time.monotonic()-started)))
   if buffer:raise ValueError('partial worker frame')
  except Exception as e:failure=str(e) if isinstance(e,Stop) else type(e).__name__
  finally:
   sel.close()
   try:os.killpg(proc.pid,signal.SIGKILL)
   except ProcessLookupError:pass
   proc.wait();proc.stdin.close();proc.stdout.close()
 receipt={'returncode':proc.returncode,'hard_timeout':timed_out,'elapsed_seconds':time.monotonic()-started,'reaped':True,'requests':request_no,'failure':failure}
 put(root/'SUPERVISION.json',receipt)
 return receipt

def promote(root,mode):
 receipt=read(root/'SUPERVISION.json')
 if receipt['returncode'] or receipt['hard_timeout'] or receipt['failure']:return False
 child=root/'child';result=read(child/'RESULT.json')
 if mode in ('control','treatment'):
  from reconcile import actor_outcome
  put(root/'RECONCILIATION.json',actor_outcome(child))
 if mode=='constructor' and result['frozen']:
  f=read(child/'FROZEN.json')
  if digest(f['candidate_source'].encode())!=f['source_digest']:raise Stop('freeze digest')
  put(root/'FROZEN.json',f)
 put(root/'RESULT.json',{**result,'authority':'timely clean root promotion'})
 return True

def stopped(ledger):
 d=read(ledger);return d['stopped'] or d['unknown_exposure_usd']!='0'

def run_pair(live,*,key=None,fixture_url=None,action_limit=100,identity=None):
 if not fixture_url and action_limit!=100:raise Stop('live horizon must be100')
 live.mkdir(exist_ok=False);put(live/'CONSUMED.json',{'task':TASK,'fixture':bool(fixture_url),'identity':identity,'order':['constructor','control','treatment'],'retry':False})
 ledger=live/'ledger.json';initial_ledger(ledger);prefix=live/'prefix/child/prefix.jsonl';freeze=live/'constructor/FROZEN.json'
 try:
  receipt=supervise('prefix',live/'prefix','-','-')
  if not promote(live/'prefix','prefix'):raise Stop('fixed prefix failed; no reselection')
  for phase in ('constructor','control','treatment'):
   # The phase clock starts before child initialization. Constructor fixed before both actors.
   deadline=time.monotonic()+CAPS[phase]['soft']
   t=Transport(phase,live/f'{phase}-raw',ledger,deadline,key=key,fixture_url=fixture_url)
   receipt=supervise(phase,live/phase,prefix,freeze if freeze.exists() else '-',t,limit=action_limit)
   okay=promote(live/phase,phase)
   if stopped(ledger):raise Stop('provider/accounting stop; no more spend')
   if phase=='constructor':
    counts=read(ledger)['arms']['constructor']
    if counts['requests'] and counts['reasoning_tokens']==0:raise Stop('constructor reasoning on not demonstrated')
    if not okay:
     put(live/phase/'RESULT.json',{'status':'CONSTRUCTOR_FAILED','frozen':False,'authority':'no promotion; child forensic only'})
   elif not okay:raise Stop('actor did not complete cleanly; incomplete matched exposure')
  a,b=[read(live/p/'RESULT.json') for p in ('control','treatment')]
  if hash_file(live/'control/child/INITIAL-EVIDENCE.json')!=hash_file(live/'treatment/child/INITIAL-EVIDENCE.json'):raise Stop('initial evidence parity')
  complete=all(r['complete_exposure'] for r in (a,b))
  levels=[r['observation']['levels_completed'] for r in (a,b)]
  interpretation='INCOMPLETE_MATCHED_EXPOSURE' if not complete else ('INCONCLUSIVE_TIED_PRIMARY' if levels[0]==levels[1] else 'DESCRIPTIVE_SINGLE_PAIR_DIFFERENCE_NO_GENERAL_EFFICACY')
  result={'official_scores':[read(live/a/'RECONCILIATION.json')['official_score'] for a in ('control','treatment')],'status':'PAIR_FINISHED','interpretation':interpretation,'primary_levels':levels,'official_status':[r['observation']['state'] for r in (a,b)],'constructor':read(live/'constructor/RESULT.json'),'arms':{'control':a,'treatment':b},'ledger':read(ledger),'fixture':bool(fixture_url),'allowed_next_action':'independent outcome review; no automatic additional run'}
  put(live/'RESULT.json',result);return result
 except Exception as e:
  result={'status':'HOLD','error':str(e) if isinstance(e,Stop) else type(e).__name__,'ledger':read(ledger),'fixture':bool(fixture_url),'allowed_next_action':'read-only audit; no retry'}
  put(live/'RESULT.json',result);raise

def main():
 p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--review',type=Path,required=True);p.add_argument('--release',type=Path,required=True);p.add_argument('--credential-approval',type=Path,required=True);args=p.parse_args()
 if not args.execute:raise Stop('--execute required')
 if (HERE/'live').exists():raise Stop('partial/consumed live exists; no retry')
 sha=gates(args.review,args.release,args.credential_approval)
 # No other module loads credentials; workers receive a strict new environment.
 from auth import load_key
 key,mechanism=load_key(credential_file_authorized=True)
 if not key:raise Stop('credentials unavailable')
 result=run_pair(HERE/'live',key=key,identity={'inventory_sha256':sha,'review_sha256':hash_file(args.review),'release_sha256':hash_file(args.release),'credential_approval_sha256':hash_file(args.credential_approval),'auth_mechanism':mechanism})
 print(json.dumps({'status':result['status'],'result':str(HERE/'live/RESULT.json'),'known_cost_usd':result['ledger']['known_cost_usd'],'unknown_exposure_usd':result['ledger']['unknown_exposure_usd']}))

if __name__=='__main__':
 try:main()
 except Exception as e:print('BLOCKED:',str(e) if isinstance(e,Stop) else type(e).__name__);sys.exit(2)
