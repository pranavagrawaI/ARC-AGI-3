"""Fail-closed exact-byte one-push gate. Check never calls Kaggle.
Authorization and independent review are external, never minted by this module.
"""
import argparse,hashlib,json,os,re,subprocess,sys
from pathlib import Path
RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[1]
PACKAGE=ROOT/'deployment/kaggle/submission-20260917-actor-critic-competition-v1'
IDENTITY='arc3-ac-competition-20260917-v1'
TARGET='pranavagrawai/'+IDENTITY
SDK='/home/pranav/.local/share/uv/tools/kaggle/bin/python'
CLI='/home/pranav/.local/bin/kaggle'
COMMAND=[CLI,'kernels','push','-p',str(PACKAGE),'--timeout','32400','--accelerator','NvidiaRtxPro6000']

class Hold(ValueError):pass

def require(ok,why):
    if not ok:raise Hold(why)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def pairs(items):
    d={}
    for k,v in items:
        require(k not in d,'Duplicate JSON key');d[k]=v
    return d

def read(path):
    d=json.loads(Path(path).read_text(),object_pairs_hook=pairs,parse_constant=lambda x: (_ for _ in ()).throw(Hold('Nonfinite JSON')))
    require(type(d) is dict and bool(d),'Nonempty JSON object required');return d

def equal(actual,expected):
    require(type(actual) is dict,'Object required')
    for k,v in expected.items():require(k in actual and type(actual[k]) is type(v) and actual[k]==v,'Wrong binding: '+k)

def bindings(m):
    return dict(identity=IDENTITY,target=TARGET,manifest_sha256=sha(RUN/'manifest.json'),card_sha256=m['card_sha256'],notebook_sha256=m['notebook_sha256'])

def reconcile(run):
    require(str(run)==str(RUN) and RUN.resolve()==RUN,'Wrong run directory')
    require(not os.path.lexists(RUN/'CONSUMED.json'),'Consumed authorization')
    m=read(RUN/'manifest.json')
    equal(m,dict(schema=1,identity=IDENTITY,target=TARGET,run_dir=str(RUN),package_dir=str(PACKAGE),command=COMMAND))
    require(type(m.get('sources')) is dict and m['sources'],'Missing sources')
    covered=set()
    for name,digest in m['sources'].items():
        p=ROOT/name
        require(type(digest) is str and re.fullmatch('[0-9a-f]{64}',digest),'Bad digest')
        require(not Path(name).is_absolute() and '..' not in Path(name).parts and p.resolve()==p,'Unsafe source')
        require(p.is_file() and sha(p)==digest,'Source mismatch: '+name);covered.add(p)
    require({p for p in PACKAGE.rglob('*') if p.is_file()} <= covered,'Unmanifested package file')
    require(all(not p.is_symlink() for p in PACKAGE.rglob('*')),'Package symlink')
    require({RUN/'launch_control.py',RUN/'EXPERIMENT-CARD.md',RUN/'test_gate.py'} <= covered,'Missing gate/card/test binding')
    require(sha(RUN/'EXPERIMENT-CARD.md')==m['card_sha256'],'Card drift')
    require(sha(PACKAGE/'submission.ipynb')==m['notebook_sha256'],'Notebook drift')
    meta=read(PACKAGE/'kernel-metadata.json');equal(meta,m['metadata'])
    require(meta['id']==TARGET and meta['title']==IDENTITY,'Title/id identity mismatch')
    normalized=subprocess.check_output([SDK,'-B','-c','from slugify import slugify;import sys;print(slugify(sys.argv[1]))',meta['title']],text=True).strip()
    require(normalized==TARGET.split('/')[1],'Kaggle SDK slug mismatch')
    # Independently rebuild in memory from allowlisted source and compare exact JSON.
    import importlib.util
    spec=importlib.util.spec_from_file_location('candidate_build',PACKAGE/'build.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
    require(json.loads((PACKAGE/'submission.ipynb').read_text())==b.notebook(),'Altered notebook payload')
    require({k:hashlib.sha256(v).hexdigest() for k,v in b.payloads().items()}==read(PACKAGE/'ASSETS.json'),'Asset mismatch')
    return m

def validate(run):
    m=reconcile(run);bind=bindings(m)
    equal(read(RUN/'final-review.json'),dict(**bind,status='APPROVED',independent=True))
    equal(read(RUN/'authorization.json'),dict(**bind,status='AUTHORIZED',max_pushes=1,max_submissions=1,used=False,command=COMMAND,run_dir=str(RUN),timeout_seconds=32400))
    return m

def push(run):
    m=validate(run)
    # O_EXCL consumes before provider invocation, including uncertain/failing calls.
    path=RUN/'CONSUMED.json'
    fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    with os.fdopen(fd,'w') as f:
        json.dump(dict(**bindings(m),command=COMMAND),f);f.flush();os.fsync(f.fileno())
    fd=os.open(RUN,os.O_RDONLY);os.fsync(fd);os.close(fd)
    r=subprocess.run(COMMAND,text=True,capture_output=True,timeout=180)
    with (RUN/'push-receipt.json').open('x') as f:json.dump(dict(command=COMMAND,stdout=r.stdout,stderr=r.stderr,returncode=r.returncode),f,indent=2)
    return r.returncode

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('action',choices=['check','push']);p.add_argument('--run-dir',required=True)
    a=p.parse_args(argv)
    try:
        if a.action=='push':return push(a.run_dir)
        validate(a.run_dir);print('READY');return 0
    except (Hold,OSError,ValueError,KeyError,TypeError,subprocess.SubprocessError) as e:
        print('HOLD: '+str(e));return 2

if __name__=='__main__':raise SystemExit(main())
