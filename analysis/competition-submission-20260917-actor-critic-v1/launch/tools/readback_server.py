"""Read-only exact server readback of the pushed kernel: metadata + source vs frozen bytes."""
import json,hashlib,sys,time,re
from pathlib import Path
RUN=Path(__file__).resolve().parents[2];ROOT=RUN.parents[1];PKG=ROOT/'deployment/kaggle/submission-20260917-actor-critic-competition-v1'
OWNER='pranavagrawai';SLUG='arc3-ac-competition-20260917-v1';VERSION=int(sys.argv[1])
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
api=KaggleApi();api.authenticate()
r=ApiGetKernelRequest();r.user_name=OWNER;r.kernel_slug=SLUG
with api.build_kaggle_client() as c:resp=c.kernels.kernels_api_client.get_kernel(r)
m=resp.metadata;b=resp.blob
meta=dict(ref=m.ref,slug=m.slug,title=m.title,author=m.author,id=m.id,current_version_number=m.current_version_number,is_private=m.is_private,enable_gpu=m.enable_gpu,enable_tpu=m.enable_tpu,enable_internet=m.enable_internet,machine_shape=m.machine_shape,docker_image=m.docker_image,kernel_type=m.kernel_type,language=m.language,competition_data_sources=list(m.competition_data_sources),dataset_data_sources=list(m.dataset_data_sources),kernel_data_sources=list(m.kernel_data_sources),model_data_sources=list(m.model_data_sources),category_ids=list(m.category_ids),last_run_time=str(m.last_run_time))
local=json.loads((PKG/'kernel-metadata.json').read_text())
fails=[]
def chk(name,a,e):
    if a!=e:fails.append(f'{name}: server={a!r} local={e!r}')
chk('ref',meta['ref'],local['id']);chk('slug',meta['slug'],SLUG);chk('title',meta['title'],local['title']);chk('ref_owner',meta['ref'].split('/')[0],OWNER)
chk('version',meta['current_version_number'],VERSION);chk('is_private',meta['is_private'],True);chk('enable_gpu',meta['enable_gpu'],True);chk('enable_tpu',meta['enable_tpu'],False);chk('enable_internet',meta['enable_internet'],False)
chk('machine_shape',meta['machine_shape'],'NvidiaRtxPro6000');chk('docker_image',meta['docker_image'],local['docker_image']);chk('kernel_type',meta['kernel_type'],'notebook');chk('language',meta['language'],'python')
chk('competition_sources',meta['competition_data_sources'],local['competition_sources']);chk('dataset_sources',sorted(meta['dataset_data_sources']),sorted(local['dataset_sources']));chk('kernel_sources',meta['kernel_data_sources'],[]);chk('model_sources',meta['model_data_sources'],[])
# source: provider may reserialize; compare parsed notebook and full executable bytes
raw=b.source;raw_sha=hashlib.sha256(raw.encode()).hexdigest()
frozen_raw=(PKG/'submission.ipynb').read_bytes();frozen_sha=hashlib.sha256(frozen_raw).hexdigest()
try:
    srv=json.loads(raw);loc=json.loads(frozen_raw)
    def cells(nb):return [(c['cell_type'],''.join(c['source']) if isinstance(c['source'],list) else c['source']) for c in nb['cells']]
    sc,lc=cells(srv),cells(loc)
    chk('cell_count',len(sc),len(lc))
    for i,(a,e) in enumerate(zip(sc,lc)):chk(f'cell{i}.type',a[0],e[0]);chk(f'cell{i}.source_sha256',hashlib.sha256(a[1].encode()).hexdigest(),hashlib.sha256(e[1].encode()).hexdigest())
    chk('parsed_equal',srv==loc,True)
    exec_sha=dict(server=hashlib.sha256(''.join(s for t,s in sc if t=='code').encode()).hexdigest(),local=hashlib.sha256(''.join(s for t,s in lc if t=='code').encode()).hexdigest())
except Exception as e:fails.append('source parse: '+str(e));exec_sha=None
out=dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),target=OWNER+'/'+SLUG,version=VERSION,server_metadata=meta,server_source_sha256=raw_sha,server_source_bytes=len(raw.encode()),frozen_notebook_sha256=frozen_sha,frozen_notebook_bytes=len(frozen_raw),raw_bytes_identical=raw.encode()==frozen_raw,executable_code_sha256=exec_sha,failures=fails,verdict='PASS' if not fails else 'FAIL')
p=RUN/'launch'/f'server-readback-v{VERSION}.json';p.write_text(json.dumps(out,indent=2)+'\n')
print(out['verdict'],json.dumps(fails),'raw_identical=',out['raw_bytes_identical'],'parsed_equal' ,exec_sha)
