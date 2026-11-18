"""One read-only status/quota/log poll; no cancellation, upload or execution."""
import datetime as dt,json,sys
from pathlib import Path
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent))
from http_error_capture import exception_record
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
api=KaggleApi();api.authenticate();ref='pranavagrawai/serving-wire-capacity9-20261023'
now=dt.datetime.now(dt.timezone.utc);stamp=now.strftime('%Y%m%dT%H%M%SZ');out={'utc':now.isoformat(),'kernel_ref':ref,'accepted_version':1,'kernel_id':140375737}
try:
    status=api.kernels_status(ref);out['status']=type(status).to_dict(status)
except Exception as e:out['status_error']=exception_record(e)
try:
    with api.build_kaggle_client() as c:q=c.kernels.kernels_api_client.get_accelerator_quota_statistics()
    out['quota']={'utc':dt.datetime.now(dt.timezone.utc).isoformat(),'used_s':q.gpu_quota.time_used.total_seconds(),'reserved_s':q.gpu_quota.time_reserved.total_seconds(),'delta_used_s':q.gpu_quota.time_used.total_seconds()-61844.901}
except Exception as e:out['quota_error']=exception_record(e)
try:
    with api.build_kaggle_client() as c:
        request=ApiListKernelSessionOutputRequest();request.user_name='pranavagrawai';request.kernel_slug='serving-wire-capacity9-20261023';request.page_size=100
        response=c.kernels.kernels_api_client.list_kernel_session_output(request)
    out['output_files']=[x.file_name for x in response.files or []];out['next_page_token']=response.next_page_token
    log=response.log or '';out['log_bytes']=len(log.encode());out['log_artifact']='polls/'+stamp+'.log'
    (P/'polls'/f'{stamp}.log').write_text(log)
except Exception as e:out['output_error']=exception_record(e)
(P/'polls'/f'{stamp}.json').write_text(json.dumps(out,indent=2)+'\n')
(P/'latest-poll.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='output_files'},indent=2))
print('output_files',len(out.get('output_files',[])))
