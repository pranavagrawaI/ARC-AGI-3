"""Terminal collection bound to accepted kernel id/version; every download pins version_number."""
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path
import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest,ApiListKernelSessionOutputRequest,ApiDownloadKernelOutputRequest

ROOT=Path(__file__).resolve().parents[2]
def main():
    original=requests.Session.send
    def send(s,r,**kw):kw['timeout']=(15,60);return original(s,r,**kw)
    requests.Session.send=send
    d=ROOT/'a5b';r=json.loads((d/'launch-response.json').read_text());m=json.loads((d/'kernel/kernel-metadata.json').read_text());owner,slug=m['id'].split('/')
    a=KaggleApi();a.authenticate()
    def bind():
        req=ApiGetKernelRequest();req.user_name=owner;req.kernel_slug=slug
        with a.build_kaggle_client() as c:remote=c.kernels.kernels_api_client.get_kernel(req)
        assert remote.metadata.id==r['kernelId'] and remote.metadata.current_version_number==r['versionNumber']
    bind();status=json.loads(str(a.kernels_status(m['id'])))
    assert status['status'] in ['COMPLETE','ERROR','CANCELED','CANCELLED'],status
    out=d/'output';out.mkdir(exist_ok=True);token=None;files=[];log=None
    while True:
        req=ApiListKernelSessionOutputRequest();req.user_name=owner;req.kernel_slug=slug;req.page_size=200
        if token:req.page_token=token
        with a.build_kaggle_client() as c:resp=c.kernels.kernels_api_client.list_kernel_session_output(req)
        files.extend(item.file_name for item in resp.files or []);log=resp.log or log;token=resp.next_page_token
        if not token:break
    manifest=[]
    if log:
        (out/'kernel-log.raw.json').write_text(log)
        try:
            parsed=json.loads(log);text=''.join(x.get('data','') for x in parsed)
        except (json.JSONDecodeError,TypeError):text=log
        (out/'kernel.log').write_text(text)
    for name in files:
        if not (name.endswith(('.json','.jsonl','.txt','.log','.parquet')) and not name.startswith(('movies/','prompts/','solver_analysis/'))):continue
        dest=(out/name).resolve();assert dest.is_relative_to(out.resolve());dest.parent.mkdir(parents=True,exist_ok=True)
        req=ApiDownloadKernelOutputRequest();req.owner_slug=owner;req.kernel_slug=slug;req.version_number=r['versionNumber'];req.file_path=name
        with a.build_kaggle_client() as c:
            response=c.kernels.kernels_api_client.download_kernel_output(req)
            with response:
                response.raise_for_status();sha=hashlib.sha256();size=0
                with dest.open('wb') as f:
                    for chunk in response.iter_content(1024*1024):sha.update(chunk);f.write(chunk);size+=len(chunk)
        manifest.append({'file':name,'bytes':size,'sha256':sha.hexdigest(),'version_number':r['versionNumber']})
    bind()
    result={'utc':dt.datetime.now(dt.timezone.utc).isoformat(),'kernel':m['id'],'kernel_id':r['kernelId'],'version':r['versionNumber'],**status,
            'download_version_number':r['versionNumber'],'files_listed':len(files),'downloaded':manifest,'hard_gpu_cap_s':6300,
            'bound_current_identity_before_and_after':True}
    (d/'collection.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='downloaded'},indent=2))

if __name__=='__main__':main()
