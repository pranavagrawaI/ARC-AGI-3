"""Terminal receipts bound to accepted id/version, using signed current-session links."""
import datetime as dt
import hashlib
import json
import sys
import os
import socket
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse
from pathlib import Path
import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest,ApiListKernelSessionOutputRequest

ROOT=Path(__file__).resolve().parent
def main():
    fallback=os.environ.get('KAGGLE_OUTPUT_IPV4')
    if fallback:
        original_dns=socket.getaddrinfo
        def resolve(host,*args,**kw):
            try:return original_dns(host,*args,**kw)
            except socket.gaierror:
                if host!='www.kaggleusercontent.com':raise
                return original_dns(fallback,*args,**kw)
        socket.getaddrinfo=resolve  # URL, SNI and certificate hostname remain unchanged.
    original=requests.Session.send
    def send(s,r,**kw):kw['timeout']=(15,45);kw['stream']=True;return original(s,r,**kw)
    requests.Session.send=send
    d=ROOT;r=json.loads((d/'evidence/launch-response.json').read_text());m=json.loads((d/'kernel/kernel-metadata.json').read_text());owner,slug=m['id'].split('/')
    a=KaggleApi();a.authenticate()
    def bind():
        req=ApiGetKernelRequest();req.user_name=owner;req.kernel_slug=slug
        with a.build_kaggle_client() as c:remote=c.kernels.kernels_api_client.get_kernel(req)
        assert remote.metadata.id==r['kernelId'] and remote.metadata.current_version_number==r['versionNumber']
    bind();status=json.loads(str(a.kernels_status(m['id'])))
    assert status['status'] in ['COMPLETE','ERROR','CANCELED','CANCELLED'],status
    out=d/'output';out.mkdir(exist_ok=True);token=None;files=[];links={};log=None
    while True:
        req=ApiListKernelSessionOutputRequest();req.user_name=owner;req.kernel_slug=slug;req.page_size=200
        if token:req.page_token=token
        with a.build_kaggle_client() as c:resp=c.kernels.kernels_api_client.list_kernel_session_output(req)
        files.extend(item.file_name for item in resp.files or [])
        for item in resp.files or []:
            if urlparse(item.url).hostname!='www.kaggleusercontent.com':raise ValueError('unexpected output host')
            links[item.file_name]=item.url
        log=resp.log or log;token=resp.next_page_token
        if not token:break
    manifest=[]
    if log:
        (out/'kernel-log.raw.json').write_text(log)
        try:
            parsed=json.loads(log);text=''.join(x.get('data','') for x in parsed)
        except (json.JSONDecodeError,TypeError):text=log
        (out/'kernel.log').write_text(text)
    core_names=('results.json','result.json','matrix.json','server-info.json','source-receipts.json','serve.log')
    files.sort(key=lambda name:(Path(name).name not in core_names,name))
    (d/'output-files.json').write_text(json.dumps(files,indent=2)+'\n')
    selected=[name for name in files if name.endswith(('.json','.jsonl','.txt','.log','.parquet'))
              and not any(part in name.split('/') for part in ['movies','prompts','solver_analysis'])]
    cache={v['file']:v for v in json.loads((d/'collection-progress.json').read_text())} if (d/'collection-progress.json').exists() else {}
    # Prior workers atomically rename only after complete TLS-verified reads from
    # version1; recover receipts lost when another future caused batch exit.
    for name in selected:
        existing=(out/name).resolve()
        if name not in cache and existing.is_file():
            cache[name]={'file':name,'bytes':existing.stat().st_size,'sha256':hashlib.sha256(existing.read_bytes()).hexdigest(),
                         'version_number':r['versionNumber'],'recovered_from_atomic_version_bound_writer':True}
    def download(name):
        dest=(out/name).resolve();assert dest.is_relative_to(out.resolve());dest.parent.mkdir(parents=True,exist_ok=True)
        previous=cache.get(name)
        if previous and previous['version_number']==r['versionNumber'] and dest.is_file() and dest.stat().st_size==previous['bytes'] and hashlib.sha256(dest.read_bytes()).hexdigest()==previous['sha256']:
            return previous
        for attempt in range(3):
            partial=dest.with_suffix(dest.suffix+f'.{time.time_ns()}.partial')
            try:
                # ListKernelSessionOutput already supplies signed CDN links.
                # Avoid one rate-limited DownloadKernelOutput RPC per file.
                with requests.get(links[name],stream=True,timeout=(10,30)) as response:
                    response.raise_for_status();sha=hashlib.sha256();size=0
                    with partial.open('wb') as f:
                        for chunk in response.iter_content(1024*1024):sha.update(chunk);f.write(chunk);size+=len(chunk)
                partial.replace(dest)
                return {'file':name,'bytes':size,'sha256':sha.hexdigest(),'version_number':r['versionNumber'],
                        'binding':'current-session signed link; id/version bound before and after'}
            except requests.RequestException as exc:
                print('read-only file retry',name,type(exc).__name__,attempt+1,flush=True)
                if attempt==2:raise
                time.sleep(1)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures={pool.submit(download,name):name for name in selected}
        for future in as_completed(futures):
            item=future.result();manifest.append(item)
            (d/'collection-progress.json').write_text(json.dumps(manifest,indent=2)+'\n')
            if len(manifest)%20==0:print('collected',len(manifest),'/',len(selected),flush=True)
    bind()
    result={'utc':dt.datetime.now(dt.timezone.utc).isoformat(),'kernel':m['id'],'kernel_id':r['kernelId'],'version':r['versionNumber'],**status,
            'download_version_number':r['versionNumber'],'files_listed':len(files),'downloaded':manifest,'hard_gpu_cap_s':9000,
            'bound_current_identity_before_and_after':True,'download_scope':'JSON/JSONL/text/log/parquet receipts; movies/prompts/solver_analysis excluded by path component'}
    (d/'collection.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='downloaded'},indent=2))

if __name__=='__main__':
    try:main()
    except requests.RequestException as exc:
        print('read-only collection failed:',type(exc).__name__,'; signed output URL omitted',flush=True)
        raise SystemExit(1)
