"""Read-only current-identity binding and a bounded live-log snapshot."""
import datetime as dt
import json
import sys
import time
from pathlib import Path
import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest,ApiGetKernelSessionLogsStreamRequest

ROOT=Path(__file__).resolve().parent
def poll():
    original=requests.Session.send
    def send(session,request,**kw):kw['timeout']=(10,10);return original(session,request,**kw)
    requests.Session.send=send
    d=ROOT/'tinfield';receipt=json.loads((d/'launch-response.json').read_text());meta=json.loads((d/'kernel/kernel-metadata.json').read_text())
    api=KaggleApi();api.authenticate();q=ApiGetKernelRequest();q.user_name,q.kernel_slug=meta['id'].split('/')
    with api.build_kaggle_client() as c:r=c.kernels.kernels_api_client.get_kernel(q)
    assert r.metadata.id==receipt['kernelId'] and r.metadata.current_version_number==receipt['versionNumber']
    def sources(n):return [(c['cell_type'],''.join(c['source'])) for c in n['cells']]
    assert sources(json.loads(r.blob.source))==sources(json.loads((d/'kernel'/meta['code_file']).read_text()))
    state=json.loads(str(api.kernels_status(meta['id'])))
    row={'utc':dt.datetime.now(dt.timezone.utc).isoformat(),'kernel_id':r.metadata.id,'version':r.metadata.current_version_number,**state}
    with (d/'status-history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    (d/'status.json').write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row),flush=True)
    if '--stream' in sys.argv:
        q=ApiGetKernelSessionLogsStreamRequest();q.user_name,q.kernel_slug=meta['id'].split('/');q.wait_for_logs_url_seconds=1
        until=time.monotonic()+35;buf=[]
        try:
            with api.build_kaggle_client() as c:
                response=c.kernels.kernels_api_client.get_kernel_session_logs_stream(q)
                with response:
                    response.raise_for_status()
                    for line in response.iter_lines(chunk_size=1024,decode_unicode=True):
                        if line:buf.append(line if isinstance(line,str) else line.decode(errors='replace'))
                        if time.monotonic()>until:break
        except requests.RequestException as e:
            status=e.response.status_code if e.response is not None else None
            print('stream boundary',type(e).__name__,status,flush=True)
            if e.response is not None:print(e.response.text[:500],flush=True)
        text='\n'.join(buf)
        stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S')
        (d/'evidence'/f'live-log-{stamp}.txt').write_text(text)
        (d/'latest-kernel-log.txt').write_text(text)
        print(text[-10000:],flush=True)
    return row

if __name__=='__main__':poll()
