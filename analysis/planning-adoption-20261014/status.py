"""Read-only id/version/source binding and a bounded live log snapshot."""
import datetime as dt
import json
from pathlib import Path
import sys
import time
import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest, ApiGetKernelSessionLogsStreamRequest

HERE=Path(__file__).resolve().parent


def main():
    identity=json.loads((HERE/'identity.json').read_text())
    accepted=json.loads((HERE/'evidence/launch-response.json').read_text())
    owner,slug=identity['metadata']['id'].split('/')
    original=requests.Session.send
    def send(session,request,**kwargs):
        kwargs['timeout']=(10,15);return original(session,request,**kwargs)
    requests.Session.send=send
    api=KaggleApi();api.authenticate()
    request=ApiGetKernelRequest();request.user_name=owner;request.kernel_slug=slug
    with api.build_kaggle_client() as client:remote=client.kernels.kernels_api_client.get_kernel(request)
    assert remote.metadata.id==accepted['kernelId'] and remote.metadata.current_version_number==accepted['versionNumber']
    local=json.loads((HERE/'kernel'/identity['metadata']['code_file']).read_text())
    def cells(notebook):return [(cell['cell_type'],''.join(cell['source'])) for cell in notebook['cells']]
    assert cells(json.loads(remote.blob.source))==cells(local)
    row={'utc':dt.datetime.now(dt.timezone.utc).isoformat(),'kernel_id':accepted['kernelId'],'version':accepted['versionNumber'],
         **json.loads(str(api.kernels_status(identity['metadata']['id']))),'source_matches':True}
    with (HERE/'evidence/status-history.jsonl').open('a') as file:file.write(json.dumps(row)+'\n')
    (HERE/'evidence/status.json').write_text(json.dumps(row,indent=2)+'\n')
    print(json.dumps(row),flush=True)
    if '--stream' not in sys.argv:return
    request=ApiGetKernelSessionLogsStreamRequest();request.user_name=owner;request.kernel_slug=slug;request.wait_for_logs_url_seconds=1
    began=time.monotonic();lines=[]
    try:
        with api.build_kaggle_client() as client:response=client.kernels.kernels_api_client.get_kernel_session_logs_stream(request)
        with response:
            response.raise_for_status()
            for line in response.iter_lines(chunk_size=512,decode_unicode=True):
                if line:lines.append(line if isinstance(line,str) else line.decode(errors='replace'))
                if time.monotonic()-began>35:break
    except requests.RequestException as error:
        print('Log stream boundary',type(error).__name__,flush=True)
    text='\n'.join(lines)
    stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S')
    (HERE/'evidence'/('live-log-'+stamp+'.txt')).write_text(text)
    print(text[-4500:],flush=True)


if __name__=='__main__':main()
