import ast, base64, hashlib, importlib.util, json, os, pathlib, subprocess, sys, tempfile
R=pathlib.Path('/home/pranav/.superset/worktrees/b420fea5-43e0-4bc6-a3d0-0629350d400b/tall-humerus')
P=R/'deployment/kaggle/actor-critic-construction-kaggle-smoke'
A=R/'analysis/actor-critic-construction-kaggle-launch'
sys.dont_write_bytecode=True
# No network allowed, including accidental library calls.
def deny(event,args):
    if event in {'socket.connect','socket.bind','socket.getaddrinfo'}: raise RuntimeError('review forbids network: '+event)
sys.addaudithook(deny)
def load(name):
    s=importlib.util.spec_from_file_location(name,P/(name+'.py')); m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
nb=json.loads((P/'construction.ipynb').read_text())
code=''.join(nb['cells'][1]['source'])
with tempfile.TemporaryDirectory(prefix='independent-notebook-') as d:
    # Redirect only absolute working directory in the actual first cell.
    old=os.getcwd(); os.chdir(d)
    try:
        try: exec(compile(code.replace("Path('/kaggle/working')",'Path('+repr(d)+')'),'actual-notebook-cell-1','exec'),{})
        except Exception as e: print('CLEAN_NOTEBOOK',type(e).__name__,str(e))
        pathlib.Path(d,'local_wire.py').write_text('')
        try: exec(compile(code.replace("Path('/kaggle/working')",'Path('+repr(d)+')'),'actual-notebook-cell-1','exec'),{})
        except Exception as e: print('ASSETS_ALREADY_IN_WORKING',type(e).__name__,str(e))
    finally: os.chdir(old)
try: load('headless')
except Exception as e: print('HEADLESS_IMPORT',type(e).__name__,str(e))
wire=load('local_wire'); audit=load('audit')
response={'model':wire.LOCAL_MODEL,'choices':[{'message':{'role':'assistant','content':'ok'}}],'usage':{'prompt_tokens':2,'completion_tokens':1}}
with tempfile.TemporaryDirectory(prefix='independent-receipt-') as d:
    p=pathlib.Path(d)
    (p/'requests.jsonl').write_text(json.dumps({'call':1,'request':{'model':wire.LOCAL_MODEL,'messages':[]}})+'\n')
    (p/'responses.jsonl').write_text(json.dumps({'call':1,'response':response,'usage':wire.usage_receipt(response)})+'\n')
    print('LOCAL_RECEIPT',wire.usage_receipt(response),'ACCOUNTED',audit.request_response_accounted(p))
    response.update(model='qwen/qwen3.8-27b',provider='CoreWeave')
    (p/'responses.jsonl').write_text(json.dumps({'call':1,'response':response})+'\n')
    print('CHANGING_ONLY_IDENTITY_TO_OPENROUTER_ACCOUNTED',audit.request_response_accounted(p))
tree=ast.parse((P/'donor_bootstrap.py').read_text())
assets=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ASSETS' for t in n.targets)))
print('DONOR_ASSET_NAMES',sorted(assets))
print('EMBEDDED_CASE_SHA256',hashlib.sha256(base64.b64decode(assets['case.json'])).hexdigest())
# Decode nested constants; never execute bootstrap or model/server code.
launcher=None
for n in ast.walk(tree):
    if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='exec' and n.args and isinstance(n.args[0],ast.Constant) and isinstance(n.args[0].value,str):
        nested=ast.parse(n.args[0].value)
        for sub in ast.walk(nested):
            if isinstance(sub,ast.Call) and isinstance(sub.func,ast.Attribute) and sub.func.attr=='b64decode' and sub.args and isinstance(sub.args[0],ast.Constant):
                raw=base64.b64decode(sub.args[0].value)
                try: text=raw.decode()
                except UnicodeDecodeError: continue
                if 'def expected_launcher_solver_requests(' in text: launcher=text
assert launcher
lt=ast.parse(launcher)
fn=next(n for n in lt.body if isinstance(n,ast.FunctionDef) and n.name=='expected_launcher_solver_requests')
ns={}; exec('from __future__ import annotations\n'+ast.unparse(fn),ns)
probe=ast.parse((P/'kaggle_probe.py').read_text())
summary=next(n.value for n in ast.walk(probe) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='summary' for t in n.targets))
keys=[ast.literal_eval(k) for k in summary.keys]
print('FINAL_PROBE_KEYS',keys,'SUPERVISOR_REQUIRES_STATUS_PASSED', 'launcher_result.get("status") != "passed"' in launcher)
try: print('SUPERVISOR_ACCOUNTING',ns['expected_launcher_solver_requests'](dict.fromkeys(keys)))
except Exception as e: print('SUPERVISOR_ACCOUNTING',type(e).__name__,str(e))
print('EMBEDDED_LAUNCHER_SHA256',hashlib.sha256(launcher.encode()).hexdigest())
m=json.loads((P/'MANIFEST.json').read_text()); mismatch=[]
for name,h in {**m['assets'],**m['files']}.items():
    if hashlib.sha256((P/name).read_bytes()).hexdigest()!=h:mismatch.append(name)
print('MANIFEST_COMPLETE',m['complete'],'MISMATCH',mismatch)
print('MISSING_DEPENDENCIES',[x for x in ['convert_frame.py','test_nooa_join.py','chassis','nooa-join','tycho_nooa'] if not (P/x).exists()])
print('HASHES')
for p in sorted([*P.glob('*'),*A.glob('*.py'),A/'EXPERIMENT-CARD.md',A/'MANIFEST.json']):
    if p.is_file():print(hashlib.sha256(p.read_bytes()).hexdigest(),str(p.relative_to(R)))
