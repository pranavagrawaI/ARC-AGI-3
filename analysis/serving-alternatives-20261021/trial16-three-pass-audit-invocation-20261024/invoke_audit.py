"""Offline invocation receipt wrapper; reviewed audit CLI executes once unchanged."""
import datetime as dt,hashlib,json,os,runpy,socket,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent
AUDIT=BASE/'trial16-final-residual-transport-local/audit_three_pass.py'
ARGV=[str(AUDIT),str(BASE/'coordination-20261024/trial16-three-pass-audit-approved.json'),str(BASE/'trial16-final-residual-collected-20261024'),str(BASE/'trial16-three-pass-audit-result-20261024')]
started=time.monotonic();captured={};network=[];exception=None;code=None

def persist(name,value):
    with (HERE/name).open('x') as stream:
        json.dump(value,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())

def profile(frame,event,value):
    if event=='return' and frame.f_code.co_filename==str(AUDIT) and frame.f_code.co_name=='supplemental':
        captured['returned']=value;sys.setprofile(None)

def offline(*args,**kwargs):
    network.append('connection attempt blocked');raise RuntimeError('offline audit attempted network')

persist('COMMAND.json',dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),actual_process_argv=sys.argv,reviewed_cli_argv=ARGV,execution='runpy __main__ of unchanged reviewed audit; profile observes supplemental return only',network='blocked',scientific_function_replacement=False))
socket.socket.connect=offline;socket.socket.connect_ex=offline;socket.create_connection=offline
sys.path.insert(0,str(AUDIT.parent));sys.argv=ARGV
sys.setprofile(profile)
try:
    runpy.run_path(str(AUDIT),run_name='__main__');code=0
except SystemExit as error:code=error.code if isinstance(error.code,int) else 1
except BaseException as error:
    exception=dict(type=type(error).__name__,message=str(error));code=1
    import traceback;traceback.print_exc()
finally:sys.setprofile(None)
persist('RETURNED.json',captured)
if 'returned' not in captured or network or exception:code=1
persist('EXIT.json',dict(exit_code=code,observed_reviewed_CLI_exit=code,exception=exception,returned_value_observed='returned' in captured,network_attempts=len(network),elapsed_seconds=time.monotonic()-started))
sys.exit(code)
