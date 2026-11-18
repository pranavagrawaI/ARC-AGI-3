"""CPU-only stand-in stdout/stderr; never imports or executes a GPU runtime."""
import os
from pathlib import Path
import sys
import time

if __name__=='__main__':
    kind=sys.argv[1]
    if kind=='file':sys.stdout.buffer.write(Path(sys.argv[2]).read_bytes())
    elif kind=='big':sys.stdout.buffer.write(b'x'*(256*1024))
    elif kind=='slow':
        Path(sys.argv[2]).write_text(str(os.getpid()));print('partial',flush=True);time.sleep(20)
    elif kind=='error':print('fixture failure',file=sys.stderr);raise SystemExit(7)
    elif kind=='ordinary-descendant':
        import subprocess,json
        marker=Path(sys.argv[2]);grandchild=Path(str(marker)+'.grandchild')
        code='import os,time;from pathlib import Path;Path(%r).write_text(str(os.getpid()));time.sleep(20)'%str(grandchild)
        try:
            child=subprocess.Popen([sys.executable,'-c',code])
            outcome=dict(created=True,pid=child.pid)
        except OSError as error:outcome=dict(created=False,errno=error.errno)
        marker.write_text(json.dumps(dict(pid=os.getpid(),**outcome)))
        print(json.dumps(outcome),flush=True);time.sleep(20)
    elif kind=='process-syscalls':
        import ctypes,json,signal
        attempts={}
        for name,number,args in [('fork',57,()),('clone_process',56,(signal.SIGCHLD,0,0,0,0)),('clone3',435,(0,0))]:
            libc=ctypes.CDLL(None,use_errno=True);ctypes.set_errno(0)
            result=libc.syscall(number,*args);error=ctypes.get_errno()
            if result==0:os._exit(0) # clean up if a broken fork/clone filter lets this fixture create a process
            if result>0:os.waitpid(result,0)
            attempts[name]=dict(result=result,errno=error)
        print(json.dumps(attempts))
    elif kind=='thread':
        import threading
        result=[];worker=threading.Thread(target=lambda:result.append('thread-ok'));worker.start();worker.join();print(result[0])
    else:raise ValueError('unknown CPU fixture')
