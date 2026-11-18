"""One version-labelled first page + one bounded primary file; no pagination/retry."""
import argparse,base64,datetime as dt,hashlib,json,sys,time
from pathlib import Path
from identity import HERE,accepted
sys.path.insert(0,str(HERE))
import primary_bundle,transport
import space

def verify(raw,readback_directory):
    value=primary_bundle.verify(raw);spec=json.loads((HERE/'packet-spec.json').read_text())
    readback=json.loads((Path(readback_directory)/'READBACK.json').read_text());ref,kid,version=accepted()
    if readback.get('error') or not readback.get('source_exact') or (readback['kernel_ref'],readback['kernel_id'],readback['version'])!=(ref,kid,version):raise ValueError('primary current/source identity gap')
    if readback['status']['status'] not in ('COMPLETE','ERROR','CANCELLED','CANCELED','CANCEL_ACKNOWLEDGED'):raise ValueError('primary collection not terminal')
    metadata=json.loads((Path(readback_directory)/'metadata.json').read_text())
    if (metadata['id'],metadata['currentVersionNumber'],metadata['ref'])!=(kid,version,ref):raise ValueError('terminal metadata drift')
    source=(Path(readback_directory)/'source.ipynb').read_bytes();inner=transport.decode(source)
    attempt=json.loads((HERE/'launch-attempt.json').read_text())
    if transport.sha(inner)!=attempt['inner_notebook_sha']:raise ValueError('primary armed source mismatch')
    values=transport.literals(transport.cell_source(json.loads(inner)))
    decoded={key:base64.b64decode(item['base64'],validate=True) for key,item in value['files'].items()}
    if json.loads(decoded['config.json'])!=spec['config'] or json.loads(decoded['assignment.json'])!=values['ASSIGNMENT'] or value['run_id']!=spec['config']['run_id']:raise ValueError('primary config/assignment/run binding')
    expected={name:item['sha256'] for name,item in values['PAYLOAD'].items() if name.endswith('.py')}
    if value['source_sha256']!=expected:raise ValueError('primary runtime source hash inventory')
    result=json.loads(decoded['result.json']);events=[json.loads(line) for line in decoded['supervisor.jsonl'].splitlines()]
    terminal=[row for row in events if row['event']=='terminal']
    if len(terminal)!=1 or terminal[0]!=dict(seq=events[-1]['seq'],t=events[-1]['t'],event='terminal',**result):raise ValueError('primary result/supervisor contradiction')
    return dict(current_exact=True,result=result,missing=value['missing'],truncated=value['truncated'],
        primary_bundle_sha256=hashlib.sha256(raw).hexdigest(),full_raw_qualification=False,slot_release_claim=False)


def fetch(api,get,destination,readback_directory):
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
    destination=Path(destination);destination.mkdir(exist_ok=False)
    if not space.check(destination)['eligible']:raise ValueError('19GiB simultaneous collection reserve required; no API')
    ref,kid,version=accepted();receipt=dict(kernel_ref=ref,kernel_id=kid,version=version,complete=False,addressing='CURRENT_SESSION_REQUIRED_FILES_NOT_ARCHIVE',started_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    end=time.monotonic()+120
    (HERE/'remote-readback').mkdir(exist_ok=True)
    with (HERE/'remote-readback/primary-once.json').open('x') as stream:json.dump(dict(kernel_id=kid,version=version,destination=str(destination)),stream)
    try:
        request=ApiListKernelSessionOutputRequest();request.user_name,request.kernel_slug=ref.split('/');request.page_size=100 # Current-session addressing; exact readback before/after, no version-label or CD claim.
        with api.build_kaggle_client() as client:response=client.kernels.kernels_api_client.list_kernel_session_output(request)
        receipt['first_page_files']=[item.file_name for item in response.files or []]
        matches=[item for item in response.files or [] if item.file_name==primary_bundle.NAME]
        if len(matches)!=1:raise ValueError('one small primary absent from first page; no pagination/retry')
        count=0;target=destination/'PRIMARY.json'
        with get(matches[0].url,stream=True,timeout=(10,60),allow_redirects=False,headers={'Accept-Encoding':'identity'}) as stream,target.open('xb') as output:
            stream.raise_for_status()
            if stream.status_code!=200 or not str(stream.headers.get('content-length','')).isdigit() or int(stream.headers['content-length'])>primary_bundle.MAX_BYTES:raise ValueError('bounded primary HTTP200/Content-Length required')
            for block in stream.iter_content(chunk_size=65536):
                count+=len(block)
                if count>primary_bundle.MAX_BYTES or time.monotonic()>end:raise ValueError('primary16MiB/120s bound')
                output.write(block)
        receipt.update(verify(target.read_bytes(),readback_directory),complete=True,bytes=count)
    except BaseException as error:
        # Never persist a signed download URL or its exception message.
        receipt['error']=dict(type=type(error).__name__,reason='primary collection failed; no retry; inspect local partial and exact readback')
        raise
    finally:(destination/'PRIMARY-RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


def main():
    p=argparse.ArgumentParser();p.add_argument('readback',type=Path);p.add_argument('destination',type=Path);a=p.parse_args()
    from kaggle.api.kaggle_api_extended import KaggleApi
    import requests
    api=KaggleApi();api.authenticate();fetch(api,requests.get,a.destination,a.readback)
if __name__=='__main__':main()
