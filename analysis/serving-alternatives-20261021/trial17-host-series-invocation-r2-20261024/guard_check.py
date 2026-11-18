"""Only the two requested process-guard cases; never imports the direct driver."""
import ast,hashlib,json,os,subprocess,sys
from pathlib import Path
here=Path(__file__).resolve().parent;source=here/'direct.py'
tree=ast.parse(source.read_bytes());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='same_python_invocation')
scope=dict(Path=Path,sys=sys);exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),scope)
match=scope['same_python_invocation'];scripts={str(source)};destination=here.parent/'trial17-host-series-collected-20261024'
parent=Path('/proc')/str(os.getppid());exe=(parent/'exe').stat();args=[x.decode() for x in (parent/'cmdline').read_bytes().split(b'\0') if x]
assert Path((parent/'exe').resolve()).name=='timeout',(parent/'exe').resolve()
assert not match((exe.st_dev,exe.st_ino),args,scripts,destination)
with subprocess.Popen([sys.executable,'-u','-c','import time; print(\"READY\",flush=True); time.sleep(5)',str(source)],stdout=subprocess.PIPE,text=True) as child:
 assert child.stdout.readline().strip()=='READY'
 p=Path('/proc')/str(child.pid);x=(p/'exe').stat();a=[v.decode() for v in (p/'cmdline').read_bytes().split(b'\0') if v]
 assert match((x.st_dev,x.st_ino),a,scripts,destination)
 child.terminate();child.wait(timeout=2)
value=dict(timeout_ancestor_excluded=True,distinct_python_exact_script_argument_detected=True,tests=2,network_calls=0,GET_attempts=0,collector_called=False,direct_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
with (here/'GUARD-CHECK.json').open('x') as f:json.dump(value,f,indent=2)
print(json.dumps(value))
