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

def main():
 if any(k.endswith('API_KEY') for k in os.environ):raise RuntimeError('worker credential environment')
 def deny(*a,**k):raise RuntimeError('worker network forbidden')
 socket.socket.connect=deny;socket.create_connection=deny
 mode,out_s,prefix_s,freeze_s,limit=sys.argv[1:];out=Path(out_s);out.mkdir(exist_ok=False);os.chdir(out)
 put(out/'WORKER.json',{'mode':mode,'credential_environment':False,'network_namespace':Path('/proc/self/ns/net').readlink().as_posix(),'uid_map':Path('/proc/self/uid_map').read_text(),'time_ns':time.time_ns()})
 if mode not in ('prefix','control','treatment'):raise ValueError('no constructor in retry')
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
