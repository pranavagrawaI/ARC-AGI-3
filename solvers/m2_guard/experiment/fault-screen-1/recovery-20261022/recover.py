"""One bounded selective log recovery; never downloads output files or launches kernels."""
import datetime, hashlib, json, signal
from pathlib import Path
import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest, ApiListKernelSessionOutputRequest
HERE = Path(__file__).resolve().parent
CAP=20*1024*1024
calls=[]
original_send=requests.Session.send

def bounded_send(self, request, **kwargs):
    kwargs.update(stream=True, timeout=(10, 45))
    response=original_send(self, request, **kwargs)
    data=bytearray()
    try:
        for chunk in response.iter_content(65536):
            data.extend(chunk)
            if len(data)>CAP:
                raise RuntimeError('response exceeds 20MiB cap')
        response._content=bytes(data)
        response._content_consumed=True
        calls.append({'method':request.method,'endpoint':request.url.split('?')[0], 'status':response.status_code, 'bytes':len(data)})
        return response
    finally:
        response.close()

def timeout(signum, frame):
    raise TimeoutError('read-only operation exceeded 60s')

def call(fn, request):
    signal.alarm(60)
    try: return fn(request)
    finally: signal.alarm(0)

def save(name, obj):
    (HERE/name).write_text(json.dumps(obj,indent=2)+'\n')

requests.Session.send=bounded_send
signal.signal(signal.SIGALRM,timeout)
api=KaggleApi(); api.authenticate()
receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(), 'kernel_id':140038081,'ref':'pranavagrawai/m2-guard-fault-screen1','expected_version':1,'file_downloads':0,'calls':calls}
try:
    with api.build_kaggle_client() as client:
        service=client.kernels.kernels_api_client
        req=ApiGetKernelRequest(); req.user_name='pranavagrawai'; req.kernel_slug='m2-guard-fault-screen1'
        result=call(service.get_kernel,req)
        metadata=result.metadata
        receipt['version_before']=metadata.current_version_number
        receipt['id_before']=metadata.id
        assert metadata.id==140038081 and metadata.current_version_number==1
        source=result.blob.source
        (HERE/'source-readback.ipynb').write_text(source)
        receipt['source_sha256']=hashlib.sha256(source.encode()).hexdigest()
        req=ApiListKernelSessionOutputRequest(); req.user_name='pranavagrawai'; req.kernel_slug='m2-guard-fault-screen1'; req.version_label='1'; req.page_size=1
        result=call(service.list_kernel_session_output,req)
        log=result.log
        (HERE/'m2-guard-fault-screen1.log').write_text(log)
        receipt['log_bytes']=len(log.encode())
        receipt['log_sha256']=hashlib.sha256(log.encode()).hexdigest()
        save('inventory-first-page.json',{'version_label':'1','files':[{k:v for k,v in type(item).to_dict(item).items() if 'url' not in k.lower()} for item in result.files or []], 'has_more':bool(result.next_page_token),'complete_inventory':not bool(result.next_page_token)})
        req=ApiGetKernelRequest(); req.user_name='pranavagrawai'; req.kernel_slug='m2-guard-fault-screen1'
        result=call(service.get_kernel,req)
        receipt['version_after']=result.metadata.current_version_number
        receipt['id_after']=result.metadata.id
        assert result.metadata.id==140038081 and result.metadata.current_version_number==1
        receipt['result']='RECOVERED' if log else 'EMPTY_LOG'
except Exception as error:
    receipt['result']='ERROR'; receipt['error']=str(error)
finally:
    save('receipt.json',receipt)
print(json.dumps(receipt,indent=2))
