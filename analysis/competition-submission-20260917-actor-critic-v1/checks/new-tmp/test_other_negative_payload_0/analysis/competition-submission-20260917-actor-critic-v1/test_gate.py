import importlib.util,json,shutil,sys
from pathlib import Path
import pytest
RUN=Path(__file__).resolve().parent

def module():
    s=importlib.util.spec_from_file_location('gate_fixture',RUN/'launch_control.py');g=importlib.util.module_from_spec(s);s.loader.exec_module(g);return g

@pytest.fixture
def gate(tmp_path,monkeypatch):
    g=module();oldroot=g.ROOT;newrun=tmp_path/g.RUN.relative_to(oldroot);newpkg=tmp_path/g.PACKAGE.relative_to(oldroot)
    newrun.mkdir(parents=True);shutil.copytree(g.PACKAGE,newpkg)
    m=g.read(g.RUN/'manifest.json')
    for n in m['sources']:
        src=oldroot/n;dst=tmp_path/n
        if not dst.exists():dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
    g.ROOT=tmp_path;g.RUN=newrun;g.PACKAGE=newpkg
    g.COMMAND=[g.CLI,'kernels','push','-p',str(newpkg),'--timeout','32400','--accelerator','NvidiaRtxPro6000']
    m.update(run_dir=str(newrun),package_dir=str(newpkg),command=g.COMMAND)
    (newrun/'manifest.json').write_text(json.dumps(m))
    bind=g.bindings(m)
    (newrun/'final-review.json').write_text(json.dumps(dict(**bind,status='APPROVED',independent=True)))
    (newrun/'authorization.json').write_text(json.dumps(dict(**bind,status='AUTHORIZED',max_pushes=1,max_submissions=1,used=False,command=g.COMMAND,run_dir=str(newrun),timeout_seconds=32400)))
    calls=[];real=g.subprocess.run
    def run(args,*a,**kw):
        if args[0]==g.CLI:calls.append(args);raise AssertionError('Provider invoked on HOLD')
        return real(args,*a,**kw)
    monkeypatch.setattr(g.subprocess,'run',run)
    return g,calls

@pytest.mark.parametrize('file',['final-review.json','authorization.json'])
@pytest.mark.parametrize('bad',['missing','empty','malformed','list','duplicate','identity','target','manifest_sha256','card_sha256','notebook_sha256'])
def test_review_authorization_negative(gate,file,bad):
    g,calls=gate;p=g.RUN/file
    if bad=='missing':p.unlink()
    elif bad=='empty':p.write_text('{}')
    elif bad=='malformed':p.write_text('{')
    elif bad=='list':p.write_text('[]')
    elif bad=='duplicate':p.write_text('{"status":"APPROVED","status":"APPROVED"}')
    else:
        d=json.loads(p.read_text());d[bad]='wrong';p.write_text(json.dumps(d))
    assert g.main(['check','--run-dir',str(g.RUN)])==2
    assert not calls and not (g.RUN/'CONSUMED.json').exists()

@pytest.mark.parametrize('bad',['consumed','title','metadata','payload','source','extra','manifest','command','run','used'])
def test_other_negative(gate,bad):
    g,calls=gate
    if bad=='consumed':(g.RUN/'CONSUMED.json').write_text('{}')
    elif bad in ('title','metadata'):
        p=g.PACKAGE/'kernel-metadata.json';d=json.loads(p.read_text());d['title' if bad=='title' else 'enable_internet']='wrong';p.write_text(json.dumps(d))
    elif bad in ('payload','source'):
        p=g.PACKAGE/('submission.ipynb' if bad=='payload' else 'play.py');p.write_text(p.read_text()+' ')
    elif bad=='extra':(g.PACKAGE/'unexpected.env').write_text('x')
    elif bad in ('manifest','command'):
        p=g.RUN/'manifest.json';d=json.loads(p.read_text());d['command' if bad=='command' else 'identity']='wrong';p.write_text(json.dumps(d))
    elif bad=='used':
        p=g.RUN/'authorization.json';d=json.loads(p.read_text());d['used']=True;p.write_text(json.dumps(d))
    path=str(g.RUN)+'/' if bad=='run' else str(g.RUN)
    assert g.main(['check','--run-dir',path])==2
    assert not calls

def test_complete_fixture_validates_without_consumption(gate):
    g,calls=gate;assert g.validate(str(g.RUN))['identity']==g.IDENTITY
    assert not calls and not (g.RUN/'CONSUMED.json').exists()

def test_push_consumes_before_provider_and_never_retries(gate,monkeypatch):
    g,_=gate;real=g.subprocess.run;calls=[]
    def provider(args,*a,**kw):
        if args[0]!=g.CLI:return real(args,*a,**kw)
        assert (g.RUN/'CONSUMED.json').is_file();calls.append(args)
        return g.subprocess.CompletedProcess(args,1,'','fixture failure')
    monkeypatch.setattr(g.subprocess,'run',provider)
    assert g.push(str(g.RUN))==1
    assert g.main(['push','--run-dir',str(g.RUN)])==2
    assert len(calls)==1
