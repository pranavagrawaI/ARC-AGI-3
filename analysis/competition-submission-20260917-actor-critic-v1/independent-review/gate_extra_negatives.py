"""Extra gate negatives beyond test_gate.py, in a disposable exact-bound copy; provider call intercepted."""
import importlib.util,json,shutil,sys,tempfile,subprocess
from pathlib import Path
RUN=Path(__file__).resolve().parents[1]
def module():
    s=importlib.util.spec_from_file_location('gate_x',RUN/'launch_control.py');g=importlib.util.module_from_spec(s);s.loader.exec_module(g);return g
results={}
with tempfile.TemporaryDirectory() as d:
    tmp=Path(d);g=module();oldroot=g.ROOT
    newrun=tmp/g.RUN.relative_to(oldroot);newpkg=tmp/g.PACKAGE.relative_to(oldroot);newrun.mkdir(parents=True);shutil.copytree(g.PACKAGE,newpkg)
    m=g.read(g.RUN/'manifest.json')
    for n in m['sources']:
        dst=tmp/n
        if not dst.exists():dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(oldroot/n,dst)
    for n in g.read(g.PACKAGE/'DONORS.json'):
        dst=tmp/n
        if not dst.exists():dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(oldroot/n,dst)
    g.ROOT=tmp;g.RUN=newrun;g.PACKAGE=newpkg;g.COMMAND=[g.CLI,'kernels','push','-p',str(newpkg),'--timeout','32400','--accelerator','NvidiaRtxPro6000']
    m.update(run_dir=str(newrun),package_dir=str(newpkg),command=g.COMMAND);(newrun/'manifest.json').write_text(json.dumps(m))
    bind=g.bindings(m)
    def review(**kw):(newrun/'final-review.json').write_text(json.dumps({**bind,'status':'APPROVED','independent':True,**kw}))
    def auth(**kw):(newrun/'authorization.json').write_text(json.dumps({**bind,'status':'AUTHORIZED','max_pushes':1,'max_submissions':1,'used':False,'command':g.COMMAND,'run_dir':str(newrun),'timeout_seconds':32400,**kw}))
    calls=[]
    def intercept(args,*a,**kw):
        if args and args[0]==g.CLI:calls.append(args);raise AssertionError('provider invoked')
        return subprocess.run.__wrapped__(args,*a,**kw) if hasattr(subprocess.run,'__wrapped__') else real(args,*a,**kw)
    real=subprocess.run;g.subprocess.run=intercept
    review();auth();results['baseline_ready']=g.main(['check','--run-dir',str(newrun)])
    cases={}
    review(independent=False);cases['review_independent_false']=g.main(['check','--run-dir',str(newrun)]);review()
    review(status='approved');cases['review_status_lowercase']=g.main(['check','--run-dir',str(newrun)]);review()
    review(manifest_sha256=bind['manifest_sha256'].upper());cases['review_manifest_sha_uppercase']=g.main(['check','--run-dir',str(newrun)]);review()
    auth(max_pushes=2);cases['auth_max_pushes_2']=g.main(['check','--run-dir',str(newrun)]);auth()
    auth(timeout_seconds=3600);cases['auth_timeout_3600']=g.main(['check','--run-dir',str(newrun)]);auth()
    auth(command=g.COMMAND[:-2]);cases['auth_command_short']=g.main(['check','--run-dir',str(newrun)]);auth()
    auth(run_dir=str(newrun)+'/');cases['auth_run_dir_trailing_slash']=g.main(['check','--run-dir',str(newrun)]);auth()
    auth(used=0);cases['auth_used_int_zero']=g.main(['check','--run-dir',str(newrun)]);auth()
    mm=dict(m,package_dir=str(newpkg)+'x');(newrun/'manifest.json').write_text(json.dumps(mm));cases['manifest_package_dir_wrong']=g.main(['check','--run-dir',str(newrun)]);(newrun/'manifest.json').write_text(json.dumps(m))
    mm=dict(m);mm['metadata']=dict(m['metadata'],competition_sources=[]);(newrun/'manifest.json').write_text(json.dumps(mm));cases['manifest_metadata_no_competition']=g.main(['check','--run-dir',str(newrun)]);(newrun/'manifest.json').write_text(json.dumps(m))
    # manifest rewritten (same content, different bytes) after review was bound -> review binding fails
    (newrun/'manifest.json').write_text(json.dumps(m,indent=2));cases['manifest_reindented_after_review']=g.main(['check','--run-dir',str(newrun)]);(newrun/'manifest.json').write_text(json.dumps(m))
    sym=newpkg/'link.py';sym.symlink_to(newpkg/'budget.py');cases['package_symlink']=g.main(['check','--run-dir',str(newrun)]);sym.unlink()
    (newpkg/'runtime/__pycache__').mkdir();(newpkg/'runtime/__pycache__/x.pyc').write_bytes(b'\0');cases['package_pyc']=g.main(['check','--run-dir',str(newrun)]);shutil.rmtree(newpkg/'runtime/__pycache__')
    cases['relative_run_dir']=g.main(['check','--run-dir',str(newrun.relative_to(Path.cwd())) if str(newrun).startswith(str(Path.cwd())) else 'relative/path'])
    results['cases']=cases;results['all_hold']=all(v==2 for v in cases.values());results['provider_calls']=len(calls)
    results['final_ready_after_restore']=g.main(['check','--run-dir',str(newrun)])
print(json.dumps(results,indent=1))
