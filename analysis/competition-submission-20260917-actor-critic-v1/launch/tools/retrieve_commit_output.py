"""Read-only: list version-1 output files (names/sizes only) and retrieve allowlisted submission.parquet by exact version_number."""
import json,subprocess,sys,time,hashlib
from pathlib import Path
RUN=Path(__file__).resolve().parents[2];SDK='/home/pranav/.local/share/uv/tools/kaggle/bin/python';VERSION=sys.argv[1]
OWNER,SLUG='pranavagrawai','arc3-ac-competition-20260917-v1'
LIST=r'''
import json,sys
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
api=KaggleApi();api.authenticate();out={}
with api.build_kaggle_client() as c:
    for scope,label in (('EXACT_VERSION_LABEL',sys.argv[1]),('CURRENT_IDENTITY',None)):
        r=ApiListKernelSessionOutputRequest();r.user_name='pranavagrawai';r.kernel_slug='arc3-ac-competition-20260917-v1';r.page_size=200
        if label:r.version_label=label
        try:
            resp=c.kernels.kernels_api_client.list_kernel_session_output(r)
            out[scope]=dict(files=[dict(name=f.file_name,size=f.size) for f in resp.files],log_lines=len(resp.log or []),next_page_token=resp.next_page_token or None)
        except Exception as e:out[scope]=dict(error=type(e).__name__,detail=str(e)[:100])
print(json.dumps(out))
'''
dest=RUN/'launch'/f'retrieved-v{VERSION}';dest.mkdir(exist_ok=True)
r=subprocess.run([SDK,'-B','-c',LIST,VERSION],text=True,capture_output=True,timeout=120)
listing=json.loads(r.stdout) if r.returncode==0 else dict(error=r.stderr[-300:])
rec=dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),version=int(VERSION),listing=listing)
out=dest/'submission.parquet'
g=subprocess.run([SDK,'-B',str(RUN/'exact_version.py'),'output','--version',VERSION,'--file','submission.parquet','--destination',str(out)],text=True,capture_output=True,timeout=300)
rec['download']=dict(returncode=g.returncode,stdout=g.stdout.strip(),stderr=g.stderr.strip()[-300:])
if out.exists():
    v=subprocess.run([str(RUN.parents[1]/'solvers/tycho_nooa/.venv/bin/python'),'-c','import pandas as pd,sys;d=pd.read_parquet(sys.argv[1]);print(__import__("json").dumps(dict(rows=len(d),columns=list(d.columns),dtypes=[str(t) for t in d.dtypes],head=d.head(5).astype(str).values.tolist())))',str(out)],text=True,capture_output=True,env={'PYTHONPATH':str(RUN/'checks/local-deps'),'PATH':'/usr/bin:/bin'})
    rec['parquet']=json.loads(v.stdout) if v.returncode==0 else dict(error=v.stderr[-300:])
    rec['parquet_sha256']=hashlib.sha256(out.read_bytes()).hexdigest();rec['parquet_bytes']=out.stat().st_size
(RUN/'launch'/f'commit-output-v{VERSION}.json').write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps(rec,indent=1))
