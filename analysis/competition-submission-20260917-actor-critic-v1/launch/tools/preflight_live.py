"""T5 live read-only preflight. Never writes to the provider. Emits LIVE-PREFLIGHT.json.
Run with the installed Kaggle SDK interpreter (python -B)."""
import hashlib,json,os,re,subprocess,sys,time,importlib.util
from pathlib import Path
RUN=Path(__file__).resolve().parents[2];ROOT=RUN.parents[1]
PKG=ROOT/'deployment/kaggle/submission-20260917-actor-critic-competition-v1'
CLI='/home/pranav/.local/bin/kaggle';SDK='/home/pranav/.local/share/uv/tools/kaggle/bin/python'
COMP='arc-prize-2026-arc-agi-3';OWNER='pranavagrawai';SLUG='arc3-ac-competition-20260917-v1';TARGET=OWNER+'/'+SLUG
EXPECT=dict(manifest='dcb716bc7809ccdc41322b25e48ab4a463b6bbe15ada1dc4b1886c3af05a2da8',notebook='3f38ad9b89d7f42423b9489b0ca2ee8a562c6993ed370f68364a98d43574e5a1',card='eb8ea21645e534e7289e480b2b262179598545c78ef0f5a74de6c240a4a8d1af',chassis='9431c4196c221c4e92cfe5c275820b2f977140cab4171412318ba6166c5bafd1',review='b18ee233d241cdf254aa894cb8686c760ba49e73c2f58b86d0e1afa6343ed0be')
SECRET=re.compile(r'(?i)(authorization|bearer|token|key|secret|signature|X-Goog|GoogleAccessId|Expires=)[^\s"]*')
def scrub(s):return SECRET.sub('<REDACTED>',s) if isinstance(s,str) else s
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cli(*args,timeout=120):
    r=subprocess.run([CLI,*args],text=True,capture_output=True,timeout=timeout,env={**os.environ,'KAGGLE_USERNAME':'','KAGGLE_KEY':''} if False else None)
    return dict(argv=[CLI,*args],returncode=r.returncode,stdout=scrub(r.stdout),stderr=scrub(r.stderr))
out=dict(task='T5',stage='A-live-preflight',utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),unix=time.time(),run_dir=str(RUN),package_dir=str(PKG),failures=[])
def fail(msg):out['failures'].append(msg)
# A1 bindings
h=dict(manifest=sha(RUN/'manifest.json'),notebook=sha(PKG/'submission.ipynb'),card=sha(RUN/'EXPERIMENT-CARD.md'),review=sha(RUN/'final-review.json'),chassis=json.loads((PKG/'runtime/CHASSIS.json').read_text())['contract_sha256'],chassis_file=sha(PKG/'runtime/CHASSIS.json'))
out['hashes']=h
for k,v in EXPECT.items():
    if h[k]!=v:fail(f'hash drift {k}: {h[k]} != {v}')
for n in ('authorization.json','CONSUMED.json','push-receipt.json','SUBMISSION-AUTHORIZATION.json'):
    if os.path.lexists(RUN/n):fail('pre-existing '+n)
out['absent_before']=['authorization.json','CONSUMED.json','push-receipt.json','SUBMISSION-AUTHORIZATION.json']
spec=importlib.util.spec_from_file_location('gate',RUN/'launch_control.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
try:
    m=g.reconcile(str(RUN));bind=g.bindings(m);g.equal(g.read(RUN/'final-review.json'),dict(**bind,status='APPROVED',independent=True));out['reconcile']='OK';out['bindings']=bind;out['sources']=len(m['sources'])
except Exception as e:fail('reconcile: '+str(e));out['reconcile']='HOLD: '+str(e)
gc=subprocess.run([sys.executable,'-B',str(RUN/'launch_control.py'),'check','--run-dir',str(RUN)],text=True,capture_output=True);out['gate_check_before_authorization']=dict(returncode=gc.returncode,stdout=gc.stdout.strip())
if not(gc.returncode==2 and 'authorization.json' in gc.stdout):fail('gate check before authorization not the expected HOLD on missing authorization')
if g.COMMAND!=[CLI,'kernels','push','-p',str(PKG),'--timeout','32400','--accelerator','NvidiaRtxPro6000']:fail('gate COMMAND drift')
out['command']=g.COMMAND
# A2 live provider, read-only
out['cli_version']=cli('--version')['stdout'].strip()
out['sdk_versions']=subprocess.check_output([SDK,'-c','import kaggle,importlib.metadata as m;print(kaggle.__version__,m.version("kagglesdk"))'],text=True).strip()
lim=cli('competitions','submission-limits',COMP,'--json');out['submission_limits']=lim
try:
    n=json.loads(lim['stdout'])['numAllowedNow']
    if not(isinstance(n,int) and n>=1):fail(f'numAllowedNow={n}')
    out['numAllowedNow']=n
except Exception as e:fail('submission-limits parse: '+str(e))
subs=cli('competitions','submissions',COMP,'--format','json');out['submissions_before']=subs
if subs['returncode']!=0 or subs['stdout'].strip()!='No submissions found':fail('unexpected submissions list state')
q=cli('quota','--format','json');out['quota_before']=q
try:
    qq=json.loads(q['stdout']);gpu=[r for r in qq if r.get('resource')=='GPU'][0];out['gpu_quota_before']=gpu
    if float(str(gpu['remaining']).rstrip('h'))<9.0:fail('GPU quota below 9h')
except Exception as e:fail('quota parse: '+str(e))
cl=cli('competitions','list','--search','arc-prize-2026','--format','json');out['competition_entry']=None
try:
    ent=[c for c in json.loads(cl['stdout']) if c['ref'].endswith('/'+COMP)][0];out['competition_entry']=ent
    if not ent.get('userHasEntered'):fail('userHasEntered false')
except Exception as e:fail('competition list: '+str(e))
# target absence, calibrated
code=r'''
import json,sys
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
api=KaggleApi();api.authenticate();res={}
res['config_username']=api.config_values.get('username')
with api.build_kaggle_client() as c:
    for slug in sys.argv[1:]:
        r=ApiGetKernelRequest();r.user_name='pranavagrawai';r.kernel_slug=slug
        try:
            resp=c.kernels.kernels_api_client.get_kernel(r);res[slug]=dict(exists=True,ref=resp.metadata.ref)
        except Exception as e:res[slug]=dict(exists=False,error=type(e).__name__,detail=str(e)[:120])
print(json.dumps(res))
'''
r=subprocess.run([SDK,'-B','-c',code,'actor-critic-construction-kaggle-r9-cd82',SLUG],text=True,capture_output=True,timeout=120)
try:
    ab=json.loads(r.stdout);out['target_absence']=ab
    if ab.get('config_username')!=OWNER:fail('authenticated identity is not owner')
    if not ab['actor-critic-construction-kaggle-r9-cd82'].get('exists'):fail('calibration kernel not visible; auth ambiguous')
    if ab[SLUG].get('exists'):fail('TARGET ALREADY EXISTS')
    if '403' not in ab[SLUG].get('detail',''):fail('target absence not the calibrated 403 form: '+ab[SLUG].get('detail',''))
except Exception as e:fail('target absence probe: '+str(e)+' '+scrub(r.stderr[-300:]))
ls=cli('kernels','list','--mine','--search',SLUG,'--format','json');out['target_search']=ls
if ls['stdout'].strip()!='Not found':fail('kernel search for target slug returned results')
# A3 metadata
meta=json.loads((PKG/'kernel-metadata.json').read_text());out['kernel_metadata']=meta
want=dict(language='python',kernel_type='notebook',is_private=True,enable_gpu=True,machine_shape='NvidiaRtxPro6000',enable_internet=False,enable_tpu=False,competition_sources=[COMP],dataset_sources=['jakobbrggen/qwen3-8-27b-fp8-hf-snapshot','mahdiseddigh/kaggle-t4-stack-2026-08-28','pranavagrawai/sglang-r8l3b-orderly-shutdown-runtime','pranavagrawai/tycho-nooa-solver-wheelhouse-cp312-v11'],kernel_sources=[],model_sources=[],id=TARGET,title=SLUG,code_file='submission.ipynb')
for k,v in want.items():
    if meta.get(k)!=v:fail(f'metadata {k}: {meta.get(k)!r} != {v!r}')
if not re.fullmatch(r'gcr\.io/kaggle-private-byod/python@sha256:[0-9a-f]{64}',meta.get('docker_image','')):fail('docker image not pinned by digest')
if set(meta)!=set(want)|{'docker_image'}:fail('unexpected metadata keys: '+str(sorted(set(meta)^(set(want)|{'docker_image'}))))
slug=subprocess.check_output([SDK,'-B','-c','from slugify import slugify;import sys;print(slugify(sys.argv[1]))',meta['title']],text=True).strip();out['sdk_slugified_title']=slug
if slug!=SLUG:fail('title does not slugify to id')
# A4 CLI flag support
hp=cli('kernels','push','--help');out['cli_push_flags']=dict(timeout='--timeout' in hp['stdout'],accelerator='--accelerator' in hp['stdout'])
if not all(out['cli_push_flags'].values()):fail('CLI lacks timeout/accelerator flags')
out['gate_files_after']={n:os.path.lexists(RUN/n) for n in out['absent_before']}
out['verdict']='GO' if not out['failures'] else 'NO-GO'
(RUN/'LIVE-PREFLIGHT.json').write_text(json.dumps(out,indent=2)+'\n')
print(out['verdict'],json.dumps(out['failures']))
