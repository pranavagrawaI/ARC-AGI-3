"""Bounded offline extensions through generated entrypoints, no candidate edits."""
import ast, importlib.util, json, os, pathlib, subprocess, sys
R = pathlib.Path('/home/pranav/.superset/worktrees/b420fea5-43e0-4bc6-a3d0-0629350d400b/tall-humerus')
OUT = pathlib.Path(__file__).resolve().parent
P = R/'deployment/kaggle/actor-critic-construction-kaggle-smoke'
A = R/'analysis/actor-critic-construction-kaggle-launch'
nb=json.loads((P/'construction.ipynb').read_text())
work=OUT/'cold-entry'; work.mkdir(exist_ok=True)
ns={}
exec(compile(''.join(nb['cells'][1]['source']).replace("Path('/kaggle/working')",f'Path({str(work)!r})'),'actual-materializer','exec'),ns)
root=ns['root']
binary=work/'synthetic-nvidia-smi'
binary.write_text('#!/bin/sh\nprintf "%s\\n" "NVIDIA RTX PRO 6000 Blackwell Server Edition, GPU-01234567-0123-0123-0123-0123456789ab, 580.1, 98304 MiB"\n')
binary.chmod(0o755)
wrapper=work/'entry.py'
wrapper.write_text(f'''import sys,runpy,tempfile
sys.path.insert(0,{str(root)!r})
tempfile.tempdir={str(OUT/'tmp')!r}
def offline(event,args):
    if event in {{'socket.connect','socket.bind','socket.getaddrinfo'}}: raise RuntimeError('review forbids network')
sys.addaudithook(offline)
import accelerator
# Supply only a synthetic executable path, preserving real GPU query/parsing.
accelerator.discover.__defaults__=([{str(binary)!r}],)
runpy.run_path({str(root/'donor_bootstrap.py')!r},run_name='__main__')
''')
env={'PATH':'/opt/bin:/usr/local/bin:/usr/bin:/bin','HOME':str(root),'LANG':'C.UTF-8','PYTHONUNBUFFERED':'1','ARC_DISABLE_DOTENV':'1'}
p=subprocess.run([sys.executable,'-I','-S','-B',str(wrapper)],env=env,capture_output=True,text=True,timeout=30)
(work/'stdout').write_text(p.stdout); (work/'stderr').write_text(p.stderr)
seal=json.loads((root/'SEAL.json').read_text()); failure=json.loads((root/'BOOTSTRAP-FAILURE.json').read_text())
assert seal['status']=='passed' and seal['cases']=={'candidate':'passed','inspection':'passed'},seal
assert p.returncode!=0 and 'expected one exact provider-normalized patched wheel' in failure['error'],failure
print('DEFAULT_BOOTSTRAP_WITH_SYNTHETIC_GPU: real sealed preflight passed; actual installer rejected absent pinned wheel before install/model calls')
# Reuse only the scripted upstream driver context; perturb its synthetic server
# access counters, never candidate code or raw local response/request receipts.
spec=importlib.util.spec_from_file_location('review_fixture',A/'tests/test_runtime.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
tree=ast.parse((A/'tests/test_runtime.py').read_text())
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='test_actual_supervisor_worker_http')
fn.decorator_list=[]
class WrongCounters(ast.NodeTransformer):
    def visit_Assign(self,node):
        if any(isinstance(t,ast.Subscript) and isinstance(t.value,ast.Name) and t.value.id=='globals_' and isinstance(t.slice,ast.Constant) and t.slice.value=='wait_for_chat_completion_count' for t in node.targets):
            for sub in ast.walk(node.value):
                if isinstance(sub,ast.BinOp) and isinstance(sub.left,ast.Constant) and sub.left.value==1:
                    sub.left.value=2
        return node
program=ast.fix_missing_locations(ast.Module(body=[WrongCounters().visit(fn)],type_ignores=[]))
exec(compile(program,'fixture-wrong-server-counters','exec'),vars(m))
import pytest
counter_work=OUT/'wrong-counter';counter_work.mkdir(exist_ok=True)
with pytest.MonkeyPatch.context() as mp:
    try:
        m.test_actual_supervisor_worker_http(counter_work,mp,None,False)
    except RuntimeError as e:
        assert 'official-launcher request-count contract did not close' in str(e),str(e)
        (counter_work/'rejection.txt').write_text(str(e)+'\n')
    else: raise AssertionError('supervisor accepted deliberately wrong server access count')
result=json.loads((counter_work/'runtime-output/solver-workspaces/raw/RESULTS.json').read_text())
assert result['status']=='passed' and result['local_requests']['complete']
print('ACTUAL_SUPERVISOR_WRONG_COUNTER: real probe/worker/game passed; production final reconciliation rejected off-by-one fixture server counts')
(OUT/'extra-results.json').write_text(json.dumps({'cold_start':{'hardware':'synthetic executable, not observed GPU','seal':seal,'first_failure':failure},'counter_rejection':{'application_status':result['status'],'admitted':result['local_requests']['admitted'],'supervisor':'rejected deliberate counter mismatch'}},indent=2)+'\n')
