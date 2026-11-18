"""Installed SDK listing + bounded filtered GET; no network at import."""
import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OVERLAY=ROOT
from collector import Stop,require
sys.path[:0]=[str(OVERLAY),str(ROOT),str(ROOT/'collection_tools')]
from http_error_capture import SAFE_HEADERS,redact
import decimal_quota,reconciliation,transport
PYTHON='/home/pranav/.local/share/uv/tools/kaggle/bin/python'
from identity import accepted

def validate_readback(directory,now=None):
    directory=Path(directory);rb=json.loads((directory/'READBACK.json').read_bytes());meta=json.loads((directory/'metadata.json').read_bytes())
    require(not rb.get('error') and rb.get('source_exact') is True,'fresh readback failed')
    ref,kid,version=accepted()
    require((rb['kernel_ref'],rb['kernel_id'],rb['version'])==(ref,kid,version) and rb['status']['status'] in ('COMPLETE','ERROR','CANCELLED','CANCELED','CANCEL_ACKNOWLEDGED'),'exact terminal current identity required')
    require((meta['ref'],meta['id'],meta['currentVersionNumber'])==(ref,kid,version),'metadata/current version mismatch')
    spec=json.loads((ROOT/'packet-spec.json').read_bytes());attempt=json.loads((ROOT/'launch-attempt.json').read_bytes())
    remote=(directory/'source.ipynb').read_bytes();local=(ROOT/'kernel'/spec['metadata']['code_file']).read_bytes()
    require(transport.cell_source(json.loads(remote))==transport.cell_source(json.loads(local))
        and transport.sha(transport.decode(remote))==attempt['inner_notebook_sha']==rb['inner_sha256'],'fresh armed source mismatch')
    quota=decimal_quota.validate(rb['typed_quota'],reconciliation.load_policy(),now)
    baseline=decimal_quota.Decimal(str(reconciliation.REVIEWED_QUOTA['used_s']))
    require(baseline<=quota['used_s']<=baseline+10500 and quota['reserved_s']==0,'terminal quota outside cap or reserved')
    return dict(kernel_id=rb['kernel_id'],version=rb['version'],source_exact=True,status=rb['status']['status'],
        inner_sha256=rb['inner_sha256'],readback_sha256=hashlib.sha256((directory/'READBACK.json').read_bytes()).hexdigest(),
        wire_seconds=rb['typed_quota']['wire_seconds'])

class Remote:
    def __init__(self):
        self.api=None
    def readback(self,destination,remaining):
        try:
            result=subprocess.run([PYTHON,'-u',str(OVERLAY/'readback.py'),str(destination)],capture_output=True,timeout=remaining)
        except subprocess.TimeoutExpired:raise Stop('fresh readback exceeded cooperative remaining time') from None
        (destination/'command-output.txt').write_text(redact((result.stdout+result.stderr)[-65536:].decode(errors='replace')))
        require(result.returncode==0,'approved Decimal readback failed; no retry')
        return validate_readback(destination)
    def list_page(self,token,page_size):
        from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
        if self.api is None:
            from kaggle.api.kaggle_api_extended import KaggleApi
            self.api=KaggleApi();self.api.authenticate()
        request=ApiListKernelSessionOutputRequest();request.user_name,request.kernel_slug=accepted()[0].split('/');request.page_size=page_size
        if token:request.page_token=token
        # Exactly the installed kernels_output addressing: omit version_label.
        with self.api.build_kaggle_client() as client:
            response=client.kernels.kernels_api_client.list_kernel_session_output(request)
        return [(item.file_name,item.url) for item in response.files or []],response.next_page_token
    def download(self,url,remaining):
        import requests
        return requests.get(url,stream=True,timeout=(min(10,remaining),min(30,remaining)),
            allow_redirects=False,headers={'Accept-Encoding':'identity'})
    @staticmethod
    def error_record(error):
        # Never read an error stream or persist a signed request URL.
        record=dict(type=type(error).__name__,retry_allowed=False)
        response=getattr(error,'response',None)
        if response is None:return record|dict(http_response='UNAVAILABLE')
        record.update(status=response.status_code,headers={k:redact(str(v))[:2048] for k,v in response.headers.items() if k.lower() in SAFE_HEADERS})
        buffered=response.__dict__.get('_content')
        if isinstance(buffered,bytes):
            record.update(buffered_bytes=len(buffered),body_prefix=redact(buffered[:65536].decode(errors='replace')),body_prefix_truncated=len(buffered)>65536)
        else:record['body']='UNAVAILABLE_WITHOUT_ADDITIONAL_STREAM_READ'
        return record
