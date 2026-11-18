"""Kill only the notebook; verify real package descendants die and reap."""
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import package_execution as fixture


def children(pid):
    try: return [int(x) for x in Path(f'/proc/{pid}/task/{pid}/children').read_text().split()]
    except FileNotFoundError: return []


def tree(pid):
    result=[]
    for child in children(pid): result.append(child); result.extend(tree(child))
    return result


def argv(pid):
    try: return Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')
    except (FileNotFoundError,ProcessLookupError): return []


def parent(pid):
    lines=Path(f'/proc/{pid}/status').read_text().splitlines()
    return int(next(x.split()[1] for x in lines if x.startswith('PPid:')))


def run(output):
    output.mkdir(parents=True,exist_ok=False)
    libc=ctypes.CDLL(None); prior=ctypes.c_int()
    assert libc.prctl(37,ctypes.byref(prior),0,0,0)==0
    assert libc.prctl(36,1,0,0,0)==0
    process=None; owned=set(); report={'scope':'LOCAL_SYNTHETIC_FIXTURE_REAL_PACKAGE_SUPERVISOR_KILL','verified':False}
    try:
        with tempfile.TemporaryDirectory(prefix='package-parentkill-') as temp:
            binary,libraries=fixture.compile_fixture(Path(temp))
            notebook='/repo/deployment/kaggle/'+fixture.HERE.name+'/qualification.ipynb'
            script='import json;from pathlib import Path; exec(compile(json.loads(Path('+repr(notebook)+').read_text())["cells"][0]["source"],"exact-notebook","exec"),{"__name__":"__main__"})'
            command=fixture.fixture_command(binary,libraries,output,script)
            process=subprocess.Popen(command,stdin=subprocess.DEVNULL,close_fds=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
            deadline=time.monotonic()+10; runner=None
            while time.monotonic()<deadline and process.poll() is None:
                descendants=tree(process.pid); owned.update(descendants)
                for pid in descendants:
                    args=argv(pid)
                    if any(x.endswith('/runner.py') for x in args) and any('/candidate-wall' in x for x in args):
                        try:
                            status = Path(f'/proc/{pid}/status').read_text()
                        except FileNotFoundError:
                            continue
                        # Observe the actual dropped identity and installed
                        # no-new-privs/filter before killing its ancestors.
                        fields = dict(line.split(':',1) for line in status.splitlines() if ':' in line)
                        if fields['Uid'].split()==['100000']*4 and fields['NoNewPrivs'].strip()=='1' and fields['Seccomp'].strip()=='2':
                            runner=pid; break
                if runner: break
                time.sleep(.005)
            assert runner, 'real sealed wall runner not reached'
            assessor=parent(runner); worker=parent(assessor); notebook_pid=parent(worker)
            assert notebook_pid in owned and len({runner,assessor,worker,notebook_pid})==4
            # Only the notebook is killed. No signal is sent to the other three.
            os.kill(notebook_pid,signal.SIGKILL)
            stdout,stderr=process.communicate(timeout=5)
            (output/'supervisor.stdout').write_bytes(stdout); (output/'supervisor.stderr').write_bytes(stderr)
            statuses={}; deadline=time.monotonic()+3
            target={runner,assessor,worker}
            while time.monotonic()<deadline:
                for pid in list(target):
                    try:
                        got,status=os.waitpid(pid,os.WNOHANG)
                        if got:
                            assert os.WIFSIGNALED(status) and os.WTERMSIG(status)==signal.SIGKILL
                            statuses[str(pid)]='SIGKILL_REAPED'; target.remove(pid)
                    except ChildProcessError:
                        if not Path(f'/proc/{pid}').exists(): statuses[str(pid)]='EXITED_REAPED_BY_PARENT'; target.remove(pid)
                if not target: break
                time.sleep(.01)
            assert not target, 'package descendants survived notebook death: '+repr(target)
            report.update(verified=True,killed_only_notebook_pid=notebook_pid,worker_pid=worker,assessor_pid=assessor,runner_pid=runner,descendant_results=statuses,launcher_returncode=process.returncode)
    finally:
        if process is not None and process.poll() is None:
            process.kill(); process.communicate(timeout=5)
        for pid in owned:
            # Reap only adopted, owned children. Never signal a recycled PID.
            try:
                got,status=os.waitpid(pid,os.WNOHANG)
                if not got:
                    os.kill(pid,9); os.waitpid(pid,0)
            except (ChildProcessError,ProcessLookupError): pass
        libc.prctl(36,prior.value,0,0,0)
        (output/'VERIFIED.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

if __name__=='__main__': print(json.dumps(run(Path(sys.argv[1]).resolve()),indent=2))
