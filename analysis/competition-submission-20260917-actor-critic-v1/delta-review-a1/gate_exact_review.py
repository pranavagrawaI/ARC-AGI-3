"""Real gate and real candidate bindings; only authority reads and consumption writes are mapped to disposable review files. Provider is intercepted."""
import importlib.util,json,os,subprocess
from pathlib import Path
from unittest.mock import patch
R=Path(__file__).resolve().parents[1];O=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('exact_gate',R/'launch_control.py');g=importlib.util.module_from_spec(s);s.loader.exec_module(g)
m=g.reconcile(str(R));bind=g.bindings(m)
review=dict(**bind,status='APPROVED',independent=True,review_task='T4R',reviewer='independent Codex delta reviewer',scope='A1 exact-byte local launch readiness only; no remote or submission success',report=str(R/'DELTA-REVIEW-A1.md'),residuals=['Recommendation B unchanged: post-gameplay validators can abort a completed run.','GPU/bootstrap, real model, private gateway writer/inventory and provider timeout semantics unexercised.','No remote public-mode path; authorizer must resolve prior remote-qualification prerequisite.','Exact authorization, live preflight and exact-version submission authorization remain required.'])
(O/'proposed-final-review.json').write_text(json.dumps(review,indent=2)+'\n')
fixture=O/'exact-authority-fixture';fixture.mkdir(exist_ok=False)
auth=dict(**bind,status='AUTHORIZED',max_pushes=1,max_submissions=1,used=False,command=g.COMMAND,run_dir=str(R),timeout_seconds=32400)
(fixture/'authorization.json').write_text(json.dumps(auth));(fixture/'final-review.json').write_bytes((O/'proposed-final-review.json').read_bytes())
realread=g.read;realrun=subprocess.run;realopen=os.open;reallexists=os.path.lexists;realpathopen=Path.open
calls=[]
def read(p):
 p=Path(p)
 return realread(fixture/p.name if p in (R/'final-review.json',R/'authorization.json') else p)
def provider(argv,*a,**kw):
 if argv and argv[0]==g.CLI:
  assert argv==g.COMMAND and (fixture/'CONSUMED.json').is_file();calls.append(argv)
  return subprocess.CompletedProcess(argv,1,'','intercepted provider failure; no external call')
 return realrun(argv,*a,**kw)
def mapped_open(p,*a,**kw):return realopen(fixture/'CONSUMED.json' if isinstance(p,(str,Path)) and Path(p)==R/'CONSUMED.json' else p,*a,**kw)
def lexists(p):return reallexists(fixture/'CONSUMED.json' if Path(p)==R/'CONSUMED.json' else p)
def pathopen(p,*a,**kw):return realpathopen(fixture/'push-receipt.json' if p==R/'push-receipt.json' else p,*a,**kw)
results={}
with patch.object(g,'read',read),patch.object(subprocess,'run',provider),patch.object(os,'open',mapped_open),patch.object(os.path,'lexists',lexists),patch.object(Path,'open',pathopen):
 assert g.validate(str(R))==m
 results['exact_proposed_review_accepted']=True
 old=json.loads((R/'PRE-REPAIR-HASHES.json').read_text())
 for file,base in [('final-review.json',review),('authorization.json',auth)]:
  p=fixture/file
  for key,oldkey in [('manifest_sha256','manifest'),('notebook_sha256','notebook')]:
   d=dict(base);d[key]=old[oldkey];p.write_text(json.dumps(d));rc=g.main(['push','--run-dir',str(R)]);assert rc==2 and not calls;results[file+':stale_'+key]=rc
  for name,value in [('missing',None),('malformed','{'),('empty','{}')]:
   if value is None:p.unlink()
   else:p.write_text(value)
   rc=g.main(['push','--run-dir',str(R)]);assert rc==2 and not calls;results[file+':'+name]=rc
  p.write_text(json.dumps(base))
 assert g.validate(str(R))==m
 assert g.push(str(R))==1 and len(calls)==1
 assert g.main(['push','--run-dir',str(R)])==2 and len(calls)==1
 results.update(intercepted_provider_calls=1,actual_provider_calls=0,consumed_before_intercept=True,second_push_held=True)
assert not (R/'authorization.json').exists() and not (R/'CONSUMED.json').exists() and not (R/'final-review.json').exists()
assert json.loads((fixture/'final-review.json').read_text())==json.loads((O/'proposed-final-review.json').read_text())
(O/'gate-exact-results.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
