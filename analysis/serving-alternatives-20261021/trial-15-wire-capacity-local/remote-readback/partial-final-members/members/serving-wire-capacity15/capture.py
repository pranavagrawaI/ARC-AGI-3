"""Controller-owned synchronous query capture; receipt workers cannot launch queries."""
import json
import hashlib
import math
import os
from pathlib import Path
import selectors
import signal
import shutil
import subprocess
import sys
import time
import uuid

import artifacts
import no_processes
from core import shutdown

HERE=Path(__file__).resolve().parent
GPU_ARGV=['nvidia-smi','--query-gpu=name,memory.used,uuid','--format=csv,noheader,nounits']
APPS_ARGV=['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits']

# Only the owning controller may hold these. Failed cleanup never discards a live handle.
RETAINED={}


class Cancelled(RuntimeError):pass


def owner_identity():
    return shutdown.read_process(os.getpid())['identity']


def check_owner(expected):
    if not expected or owner_identity()!=expected:
        raise shutdown.UnsafeIdentity('query must execute in its bound long-lived controller')


def validate_containment(guard,process_key):
    if (guard.get('contract')!='NO_NEW_PROCESSES_THREADS_ALLOWED' or guard.get('filter_sha256')!=no_processes.SHA
            or guard.get('arch')!='x86_64' or guard.get('pid')!=process_key['pid']
            or guard.get('starttime')!=process_key['starttime']
            or guard.get('seccomp_mode')!=2 or guard.get('no_new_privs')!=1):
        raise ValueError('containment identity/contract mismatch')
    return True



def validate_ready_identity(initial,current,helper_argv):
    wanted=hashlib.sha256(b''.join(os.fsencode(v)+b'\0' for v in helper_argv)).hexdigest()
    if current is None:raise shutdown.UnsafeIdentity('query helper missing at ready gate')
    expected=dict(initial['identity'])
    if expected['cmdline_sha256'] not in (hashlib.sha256(b'').hexdigest(),wanted):
        raise shutdown.UnsafeIdentity('unexpected initial helper command line')
    expected['cmdline_sha256']=wanted
    if current['ppid']!=initial['ppid'] or current['identity']!=expected:
        raise shutdown.UnsafeIdentity('query ready identity/expected argv drift')
    return True


def command(intended,executed,root,end,policy='LOCAL_CPU_ONLY',*,owner,cancelled=lambda:False):
    check_owner(owner)
    if any(value['owner']==os.getpid() for value in RETAINED.values()):raise RuntimeError('unresolved controller query; no further dispatch')
    if policy=='FUTURE_REVIEWED_GPU_QUERY':
        if os.environ.get('LOCAL_ADAPTER_PROOF')=='1':raise ValueError('GPU execution forbidden during local proof')
        if intended not in (GPU_ARGV,APPS_ARGV) or executed!=intended:raise ValueError('exact GPU query allowlist')
        binary=shutil.which(executed[0])
        if binary is None:raise ValueError('GPU query executable unavailable')
        executed=[str(Path(binary).resolve()),*executed[1:]]
    elif policy!='LOCAL_CPU_ONLY' or Path(executed[0]).resolve()!=Path(sys.executable).resolve():
        raise ValueError('LOCAL CPU execution only; GPU/external command forbidden')
    start=time.monotonic()
    if type(end) not in (int,float) or not math.isfinite(end) or not 0<end-start<=600:raise ValueError('finite <=600s query absolute end required')
    shutdown.before_deadline(end)
    # Existing absolute end includes cleanup. Reserve part of the remaining time,
    # never extend it after timeout/cancellation. Kernel syscall timing is cooperative.
    work_end=end-.1 # fixed cleanup reserve; never shrink it to admit a late query
    def checkpoint():
        shutdown.before_deadline(work_end)
        check_owner(owner)
        if cancelled():raise Cancelled('query cancellation/receipt-worker death')
    checkpoint()
    files={};streams={};totals={'stdout':0,'stderr':0};complete=False;error=None
    for name in totals:files[name],streams[name]=artifacts.new_file(root,name)
    guard_path,guard_stream=artifacts.new_file(root,'containment')
    gate_read,gate_write=os.pipe2(os.O_CLOEXEC)
    child=None;pidfd=None;intent=None;process_key=None;containment=False;guard_error=None
    selector=selectors.DefaultSelector();cleanup_error=None;authorized_at=None;bound_at=None
    cleanup_started=None;cleanup_finished=None;kill_sent_at=None;query_gone=False;retained_token=None
    try:
        checkpoint()
        helper_argv=[sys.executable,str(HERE/'command_child.py'),str(os.getpid()),json.dumps(executed),str(guard_stream.fileno()),str(gate_read)]
        child=subprocess.Popen(helper_argv,
             pass_fds=(guard_stream.fileno(),gate_read),stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
        guard_stream.close();os.close(gate_read);gate_read=None
        # Child cannot exec a query until this controller owns and verifies its pidfd.
        pidfd=os.pidfd_open(child.pid)
        observed=shutdown.read_process(child.pid)
        if (observed is None or observed['ppid']!=owner['pid'] or observed['identity']['pgid']!=child.pid
                or observed['identity']['sid']!=child.pid):raise shutdown.UnsafeIdentity('query owner/session identity')
        process_key={k:observed['identity'][k] for k in ('pid','starttime','pgid','sid','boot_id')}
        bound_at=time.monotonic()
        intent=artifacts.save(root,'command-intent',json.dumps(dict(intended_argv=intended,executed_argv=executed,
             child_pid=child.pid,process_key=process_key,observed_helper=observed,helper_argv=helper_argv,controller=owner,pidfd_bound_at=bound_at,
             auxiliary_parent_death_signal='SIGKILL_NOT_CLEANUP_AUTHORITY',start=start)).encode())
        while True:
            checkpoint()
            if guard_path.stat().st_size>4096:raise ValueError('containment receipt size')
            try:guard=json.loads(guard_path.read_text())
            except json.JSONDecodeError:
                if child.poll() is not None:raise ValueError('helper exited before containment/gate')
                time.sleep(min(.002,max(0,work_end-time.monotonic())));continue
            containment=validate_containment(guard,process_key);break
        checkpoint()
        current=shutdown.read_process(child.pid)
        try:validate_ready_identity(observed,current,helper_argv)
        except shutdown.UnsafeIdentity:
            artifacts.save(root,'pre-exec-identity-drift',json.dumps(dict(before=observed,after=current)).encode())
            raise
        checkpoint()
        shutdown.before_deadline(work_end)
        os.write(gate_write,b'G');authorized_at=time.monotonic();os.close(gate_write);gate_write=None
        for name,stream in [('stdout',child.stdout),('stderr',child.stderr)]:
            os.set_blocking(stream.fileno(),False);selector.register(stream,selectors.EVENT_READ,name)
        while selector.get_map():
            checkpoint()
            for key,_ in selector.select(max(0,min(.02,work_end-time.monotonic()))):
                checkpoint();block=os.read(key.fileobj.fileno(),65536)
                if not block:selector.unregister(key.fileobj);continue
                name=key.data
                if totals[name]+len(block)>artifacts.MAX_RAW_BYTES:raise ValueError('command raw artifact cap')
                streams[name].write(block);totals[name]+=len(block)
        while child.poll() is None:
            checkpoint();time.sleep(min(.002,max(0,work_end-time.monotonic())))
        complete=True
    except BaseException as failure:
        error=repr(failure)
        if not containment:guard_error=error
    finally:
        cleanup_started=time.monotonic()
        guard_stream.close()
        for fd in (gate_read,gate_write):
            if fd is not None:os.close(fd)
        if child is not None:
            try:
                if child.poll() is None:
                    if pidfd is None:raise shutdown.UnsafeIdentity('query pidfd not bound; no numeric PID fallback')
                    shutdown.before_deadline(end)
                    signal.pidfd_send_signal(pidfd,signal.SIGKILL);kill_sent_at=time.monotonic()
            except ProcessLookupError:pass
            except BaseException as failure:cleanup_error=repr(failure)
            try:
                child.wait(timeout=max(0,end-time.monotonic()))
                query_gone=child.poll() is not None
            except BaseException as failure:cleanup_error=(cleanup_error+'; ' if cleanup_error else '')+repr(failure)
            finally:
                if child.stdout:child.stdout.close()
                if child.stderr:child.stderr.close()
        if child is not None and not query_gone:
            retained_token=uuid.uuid4().hex
            RETAINED[retained_token]=dict(owner=os.getpid(),child=child,pidfd=pidfd,absolute_deadline=end)
            pidfd=None # ownership remains with the controller; no loss of a live handle
        if pidfd is not None:os.close(pidfd)
        selector.close()
        for stream in streams.values():stream.close()
        cleanup_finished=time.monotonic()
        if child is not None and (not query_gone or cleanup_finished>end):
            cleanup_error=cleanup_error or 'query not verified cleaned within absolute deadline'
    refs={name:artifacts.reference(path,Path(root),complete,start,cleanup_finished) for name,path in files.items()}
    guard_ref=artifacts.reference(guard_path,Path(root),containment,start,cleanup_finished)
    ended=time.monotonic()
    return dict(ok=complete and child is not None and child.returncode==0 and ended<=end and cleanup_error is None and containment,
        start=start,end=ended,absolute_deadline=end,work_deadline=work_end,
        intended_argv=intended,executed_argv=executed,exit_code=child.returncode if child else None,error=error,
        artifacts=refs,intent=intent,child_pid=child.pid if child else None,child_reaped=query_gone,cleanup_error=cleanup_error,
        owner_pid=owner['pid'],pidfd_bound_at=bound_at,execution_authorized_at=authorized_at,
        cleanup_start=cleanup_started,cleanup_end=cleanup_finished,kill_sent_at=kill_sent_at,
        cleanup_verified=query_gone and cleanup_error is None and cleanup_finished<=end,retained_query=retained_token,
        containment=dict(verified=containment,artifact=guard_ref,error=guard_error,filter_sha256=no_processes.SHA),execution_scope=policy)
