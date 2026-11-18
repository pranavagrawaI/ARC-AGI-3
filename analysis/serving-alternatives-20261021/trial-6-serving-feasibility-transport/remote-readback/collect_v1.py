import datetime as dt,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from http_error_capture import exception_record
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest,ApiListKernelSessionOutputRequest
P=Path(__file__).resolve().parent
api=KaggleApi();api.authenticate();ref='pranavagrawai/serving-candidate-feas6-20261023'
out={'utc':dt.datetime.now(dt.timezone.utc).isoformat(),'kernel_ref':ref,'accepted_version':1,'accepted_kernel_id':140341756}
try:
    with api.build_kaggle_client() as c:
        request=ApiGetKernelRequest();request.user_name='pranavagrawai';request.kernel_slug='serving-candidate-feas6-20261023';request.version_label='1'
        response=c.kernels.kernels_api_client.get_kernel(request)
    (P/'get-kernel-v1.json').write_text(json.dumps(type(response).to_dict(response),indent=2)+'\n')
    (P/'source-v1.ipynb').write_text(response.blob.source)
    (P/'metadata-v1.json').write_text(json.dumps(type(response.metadata).to_dict(response.metadata),indent=2)+'\n')
    out['source_readback']='source-v1.ipynb';out['metadata']='metadata-v1.json'
except Exception as e:out['get_kernel_error']=exception_record(e)
try:
    status=api.kernels_status(ref);out['status']=type(status).to_dict(status)
except Exception as e:out['status_error']=exception_record(e)
try:
    with api.build_kaggle_client() as c:q=c.kernels.kernels_api_client.get_accelerator_quota_statistics()
    out['quota']={'utc':dt.datetime.now(dt.timezone.utc).isoformat(),'used_s':q.gpu_quota.time_used.total_seconds(),'reserved_s':q.gpu_quota.time_reserved.total_seconds()}
except Exception as e:out['quota_error']=exception_record(e)
try:
    with api.build_kaggle_client() as c:
        request=ApiListKernelSessionOutputRequest();request.user_name='pranavagrawai';request.kernel_slug='serving-candidate-feas6-20261023';request.page_size=100
        response=c.kernels.kernels_api_client.list_kernel_session_output(request)
    (P/'initial-kernel.log').write_text(response.log or '')
    out['output_listing']=[{'name':x.file_name} for x in response.files or []];out['initial_log_bytes']=len((response.log or '').encode())
except Exception as e:out['output_listing_error']=exception_record(e)
(P/'initial-status-quota.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
