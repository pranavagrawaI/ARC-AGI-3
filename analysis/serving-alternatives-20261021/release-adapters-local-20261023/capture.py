"""Bounded actual subprocess plumbing; local policy executes CPU fixtures only."""
import json
import os
from pathlib import Path
import selectors
import signal
import shutil
import subprocess
import sys
import time

import artifacts
from core import shutdown

HERE=Path(__file__).resolve().parent
GPU_ARGV=['nvidia-smi','--query-gpu=name,memory.used,uuid','--format=csv,noheader,nounits']
APPS_ARGV=['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits']


def command(intended,executed,root,end,policy='LOCAL_CPU_ONLY'):
    # A policy label is not launch authority. The remote-only branch must be bound
    # by a future reviewed runner/assignment; all local qualification denies it.
    if policy=='FUTURE_REVIEWED_GPU_QUERY':
        if os.environ.get('LOCAL_ADAPTER_PROOF')=='1':raise ValueError('GPU execution forbidden during local proof')
        if intended not in (GPU_ARGV,APPS_ARGV) or executed!=intended:raise ValueError('exact GPU query allowlist')
        binary=shutil.which(executed[0])
        if binary is None:raise ValueError('GPU query executable unavailable')
        executed=[str(Path(binary).resolve()),*executed[1:]]
    elif policy!='LOCAL_CPU_ONLY' or Path(executed[0]).resolve()!=Path(sys.executable).resolve():
        raise ValueError('LOCAL CPU execution only; GPU/external command forbidden')
    start=time.monotonic();shutdown.before_deadline(end)
    files={};streams={};totals={'stdout':0,'stderr':0};complete=False;error=None
    for name in totals:
        files[name],streams[name]=artifacts.new_file(root,name)
    child=subprocess.Popen([sys.executable,str(HERE/'command_child.py'),str(os.getpid()),json.dumps(executed)],
                           stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
    pidfd=None;intent=None;selector=selectors.DefaultSelector();cleanup_error=None
    try:
        pidfd=os.pidfd_open(child.pid)
        fields=Path(f'/proc/{child.pid}/stat').read_text().rsplit(')',1)[1].split()
        process_key=dict(pid=child.pid,starttime=int(fields[19]),pgid=int(fields[2]),sid=int(fields[3]),
                         boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
        intent=artifacts.save(root,'command-intent',json.dumps(dict(intended_argv=intended,executed_argv=executed,
            child_pid=child.pid,process_key=process_key,parent_pid=os.getpid(),parent_death_signal='SIGKILL',start=start)).encode())
        for name,stream in [('stdout',child.stdout),('stderr',child.stderr)]:
            os.set_blocking(stream.fileno(),False);selector.register(stream,selectors.EVENT_READ,name)
        while selector.get_map():
            shutdown.before_deadline(end)
            for key,_ in selector.select(max(0,end-time.monotonic())):
                shutdown.before_deadline(end)
                block=os.read(key.fileobj.fileno(),65536)
                if not block:selector.unregister(key.fileobj);continue
                name=key.data
                if totals[name]+len(block)>artifacts.MAX_RAW_BYTES:raise ValueError('command raw artifact cap')
                streams[name].write(block);totals[name]+=len(block)
        child.wait(timeout=max(0,end-time.monotonic()))
        complete=True
    except Exception as failure:error=repr(failure)
    finally:
        if child.poll() is None:
            if pidfd is None:pidfd=os.pidfd_open(child.pid)
            try:signal.pidfd_send_signal(pidfd,signal.SIGKILL)
            except ProcessLookupError:pass
        try:child.wait(timeout=.2)
        except subprocess.TimeoutExpired as failure:cleanup_error=repr(failure)
        if pidfd is not None:os.close(pidfd)
        selector.close();child.stdout.close();child.stderr.close()
        for stream in streams.values():stream.close()
    ended=time.monotonic()
    refs={name:artifacts.reference(path,Path(root),complete,start,ended) for name,path in files.items()}
    ended=time.monotonic() # hashing/receipt finalization is included in command completion
    return dict(ok=complete and child.returncode==0 and ended<=end and cleanup_error is None,start=start,end=ended,
                intended_argv=intended,executed_argv=executed,exit_code=child.returncode,error=error,
                artifacts=refs,intent=intent,child_pid=child.pid,child_reaped=child.poll() is not None,cleanup_error=cleanup_error,
                execution_scope=policy)
