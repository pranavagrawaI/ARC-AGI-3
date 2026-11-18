"""Credential-free network-isolated worker. Trusted parent owns provider transport."""
import dataclasses,json,os,socket,sys,time
from pathlib import Path
# Parent death must retire the credential-free worker and, transitively, sealed children.
import ctypes,signal
libc=ctypes.CDLL(None,use_errno=True)
if libc.prctl(1,signal.SIGKILL,0,0,0)!=0:raise RuntimeError('parent death guard')
if os.getppid()!=int(os.environ.get('UTILITY_PARENT_PID','-1')):raise RuntimeError('parent disappeared before guard')
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE/'engine')]
from common import read,put,journal,encoded,digest,hash_file

WIRE=sys.stdout.buffer
sys.stdout=sys.stderr

def relay(body):
 WIRE.write(encoded({'request':body}));WIRE.flush()
 raw=sys.stdin.buffer.readline(8_000_002)
 if len(raw)>8_000_001 or not raw.endswith(b'\n'):raise RuntimeError('parent response framing')
 payload=json.loads(raw)
 if set(payload)!={'response'}:raise RuntimeError('provider parent stopped')
 return payload['response']
relay.label='parent-owned OpenRouter pinned hosted FP8; no checkpoint equivalence'

def construct(prefix,out):
 from model import audit
 from constructor import run_construction,ConstructionLimits,ConstructionResult
 from prefix import project_transitions
 from host import bound_constructor
 started=time.monotonic();result=ConstructionResult(status='WORKING');n=0
 def progress(r):
  nonlocal n
  put(out/f'PROGRESS-{n:03}.json',dataclasses.asdict(r));n+=1
 try:
  with audit(out),bound_constructor():
   run_construction(project_transitions(prefix),transport=relay,limits=ConstructionLimits(max_model_requests=6,wall_seconds=240),event_sink=lambda e:journal(out/'tool-events.jsonl',e),progress_sink=progress,result=result)
  put(out/'CONSTRUCTION.json',dataclasses.asdict(result))
  if result.status=='CANDIDATE_READY' and time.monotonic()-started<240:
   if result.evaluation['visible']['status']!='passed' or result.evaluation['evidence_claim']!='TYPED_VISIBLE_REPLAY_ONLY_NOT_PREDICTIVE_SUPPORT':raise ValueError('typed evaluation')
   put(out/'FROZEN.json',{'schema':'typed-constructor-freeze-v1','candidate_source':result.candidate_source,'source_digest':digest(result.candidate_source.encode()),'prefix_sha256':hash_file(prefix),'evidence_claim':'TYPED_VISIBLE_REPLAY_ONLY_NOT_PREDICTIVE_SUPPORT','predictive_support':False})
  put(out/'RESULT.json',{'status':result.status,'frozen':(out/'FROZEN.json').is_file(),'elapsed_seconds':time.monotonic()-started})
 except Exception as e:
  put(out/'FAILURE.json',{'type':type(e).__name__,'error':str(e)[:2000]})
  put(out/'RESULT.json',{'status':'CONSTRUCTOR_FAILED','frozen':False,'elapsed_seconds':time.monotonic()-started})

def main():
 if any(k.endswith('API_KEY') for k in os.environ):raise RuntimeError('worker credential environment')
 def deny(*a,**k):raise RuntimeError('worker network forbidden')
 socket.socket.connect=deny;socket.create_connection=deny
 mode,out_s,prefix_s,freeze_s,limit=sys.argv[1:];out=Path(out_s);out.mkdir(exist_ok=False);os.chdir(out)
 put(out/'WORKER.json',{'mode':mode,'credential_environment':False,'network_namespace':Path('/proc/self/ns/net').readlink().as_posix(),'uid_map':Path('/proc/self/uid_map').read_text(),'time_ns':time.time_ns()})
 if mode=='constructor':construct(Path(prefix_s),out);return
 from game import Game
 game=Game(out)
 try:
  if mode=='prefix':put(out/'RESULT.json',{'prefix':game.prefix()})
  else:
   from actor import run
   from model import audit
   expected=read(Path(prefix_s).parent/'PREFIX.json')
   with audit(out):run(game,mode,freeze_s if mode=='treatment' and freeze_s!='-' else None,relay,out,expected,action_limit=int(limit))
   put(out/'RESULT.json',read(out/'ACTOR-RESULT.json'))
 finally:game.close()

if __name__=='__main__':
 try:main()
 except Exception as e:
  # Only worker errors; no credential or request object exists here.
  print(type(e).__name__+': '+str(e)[:2000],file=sys.stderr);sys.exit(2)
