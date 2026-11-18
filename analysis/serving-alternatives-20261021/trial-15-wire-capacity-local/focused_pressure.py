"""Each case runs in a separate <=600s CPU process; no GPU or external network."""
import argparse,fcntl,hashlib,json,os,resource,signal,struct,subprocess,sys,time,traceback
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('variant',choices=['packet']);p.add_argument('case',choices=['pressure','direct_low','direct_high','write_high']);p.add_argument('output',type=Path);a=p.parse_args()
out=a.output.resolve();out.mkdir(exist_ok=False);signal.alarm(590);sys.path.insert(0,str(ROOT))
import phase_tools,probe_ipc
from core import shutdown
result=dict(variant=a.variant,case=a.case,started=time.monotonic(),gpu_work=False,external_network=0,source_hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ('phase_tools.py','probe_ipc.py','controller_shutdown.py')})
handles=[];fds=[];children=[];tools=None;native_handles=[]
def fd_state():
 values=[int(p.name) for p in Path('/proc/self/fd').iterdir() if p.name.isdigit()];return dict(count=len(values),maximum=max(values))
def record_select(read,write,error,timeout=None):
 values=[v if isinstance(v,int) else v.fileno() for v in list(read)+list(write)+list(error)]
 result.setdefault('select_calls',[]).append(dict(fds=values,caller=traceback.extract_stack(limit=4)[-2].name))
 return original_select(read,write,error,timeout)
original_select=probe_ipc.select.select
try:
    if a.case!='pressure':
        r,w=os.pipe2(os.O_NONBLOCK|os.O_CLOEXEC);fds.extend([r,w])
        if a.case in ('direct_high','write_high'):
            index=1 if a.case=='write_high' else 0;high=fcntl.fcntl((r,w)[index],fcntl.F_DUPFD_CLOEXEC,2048);fds.append(high)
            if index:w=high
            else:r=high
        end=time.monotonic()+.05;result.update(read_fd=r,write_fd=w,absolute_end=end)
        with patch.object(probe_ipc.select,'select',side_effect=record_select):
            if a.case=='write_high':
                while True:
                    try:os.write(w,b'x'*4096)
                    except BlockingIOError:break
                probe_ipc.write_frame(w,{'eligible':True},end)
            else:
                probe_ipc.write_frame(w,{'eligible':True},end);result['frame']=probe_ipc.read_frame(r,end)
        result['operation_completed']=True
    else:
        config=json.loads((ROOT/'packet-spec.json').read_text())['config'];config.update(mode='CPU_FIXTURE_ONLY',cwd=str(ROOT),run_id='CPU_DESCRIPTOR_PRESSURE',total_seconds=550,cleanup_seconds=10)
        tools=phase_tools.PhaseTools(config,out/'phase');end=tools.deadline(140)
        # Retain native-fixture pidfds BEFORE pressure. They remain cleanup authority.
        ready=out/'frontend.json';wrapper=subprocess.Popen([sys.executable,str(ROOT/'tests/pressure_fixture.py'),'wrapper',str(ready)],start_new_session=True);children.append(wrapper)
        while not ready.exists():
            if time.monotonic()>end:raise TimeoutError('CPU frontend setup')
            time.sleep(.002)
        frontend=json.loads(ready.read_text())['pid'];argv=[sys.executable,str(ROOT/'tests/pressure_fixture.py'),'frontend',str(ready)]
        binding=shutdown.bind(wrapper,frontend,argv,{str(ROOT/'tests/pressure_fixture.py'):hashlib.sha256((ROOT/'tests/pressure_fixture.py').read_bytes()).hexdigest()})
        for pid in binding['members']:native_handles.append((pid,os.pidfd_open(pid)))
        portfile=out/'http.json';server=tools.start('cpu.http',dict(argv=[sys.executable,str(ROOT/'tests/pressure_fixture.py'),'http',str(portfile)]),end)
        while not portfile.exists():
            if server.poll() is not None or time.monotonic()>end:raise RuntimeError('CPU HTTP setup')
            time.sleep(.002)
        port=json.loads(portfile.read_text())['port'];http=dict(ready_url=f'http://127.0.0.1:{port}/health',ready_seconds=130,request_seconds=2,ready_poll_seconds=.001,ready_retry_http_statuses=[503])
        result['before_ready_fds']=fd_state();tools.ready('control',http,server,end)
        result['after_ready_fds']=fd_state();result['retained_parent_handles']=len(tools.handles);result['all_parent_handles_open']=all(not h.closed for h in tools.handles)
        result['health_files']=len(list((out/'phase').glob('control.health-*.stdout')))
        # The live server's own logs must remain open until its group completes.
        result['live_http_log_handles_open']=all(not h.closed for h in getattr(tools,'phase_handles',{}).get(server.pid,tools.handles[:2]))
        probes={name:(lambda:dict(eligible=False,CPU_ONLY=True)) for name in shutdown.PROBES}
        with patch.object(probe_ipc.select,'select',side_effect=record_select):result['observation']=shutdown.observe(binding,probes,time.monotonic()+2,1)
        result['operation_completed']=True
except BaseException as error:
    result.update(operation_completed=False,error_type=type(error).__name__,error=repr(error),traceback=traceback.format_exc())
finally:
    if tools is not None:
        try:tools.cleanup()
        except BaseException as error:result['phase_cleanup_error']=repr(error)
        for h in tools.handles:h.close()
        tools.trace.close()
    for pid,fd in native_handles:
        try:signal.pidfd_send_signal(fd,signal.SIGKILL)
        except ProcessLookupError:pass
    for child in children:
        try:child.wait(timeout=2)
        except BaseException as error:result['fixture_cleanup_error']=repr(error)
    import select
    observations=[]
    for pid,fd in native_handles:
        poll=select.poll();poll.register(fd,select.POLLIN);observations.append(dict(pid=pid,pidfd_exited=bool(poll.poll(2000))));os.close(fd)
    result['retained_fixture_cleanup']=observations
    for fd in fds:
        try:os.close(fd)
        except OSError:pass
    result.update(finished=time.monotonic(),final_fds=fd_state())
    (out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result.get(k) for k in ('variant','case','operation_completed','error_type','after_ready_fds','retained_parent_handles','health_files')}));signal.alarm(0)
