import datetime as dt,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from http_error_capture import exception_record
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
P=Path(__file__).resolve().parent;api=KaggleApi();api.authenticate();ref='pranavagrawai/serving-candidate-feas6-20261023'
out={'started_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'accepted_version':1,'kernel_id':140341756,'kernel_ref':ref}
try:
    files,token=api.kernels_output(ref,path=str(P/'output'),force=False,quiet=True,page_size=100)
    out['output_files']=files;out['output_next_page_token']=token
except Exception as e:out['output_error']=exception_record(e)
# Latest source readback is useful independently of version-label API availability.
try:
    with api.build_kaggle_client() as c:
        request=ApiGetKernelRequest();request.user_name='pranavagrawai';request.kernel_slug='serving-candidate-feas6-20261023'
        response=c.kernels.kernels_api_client.get_kernel(request)
    (P/'get-kernel-latest.json').write_text(json.dumps(type(response).to_dict(response),indent=2)+'\n')
    (P/'source-latest.ipynb').write_text(response.blob.source)
    (P/'metadata-latest.json').write_text(json.dumps(type(response.metadata).to_dict(response.metadata),indent=2)+'\n')
    out['source_readback']='source-latest.ipynb'
except Exception as e:out['latest_get_kernel_error']=exception_record(e)
try:
    s=api.kernels_status(ref);out['status']=type(s).to_dict(s)
except Exception as e:out['status_error']=exception_record(e)
with api.build_kaggle_client() as c:q=c.kernels.kernels_api_client.get_accelerator_quota_statistics()
out['quota']={'utc':dt.datetime.now(dt.timezone.utc).isoformat(),'used_s':q.gpu_quota.time_used.total_seconds(),'reserved_s':q.gpu_quota.time_reserved.total_seconds(),'delta_used_s':q.gpu_quota.time_used.total_seconds()-59265.958}
(P/'terminal-status-quota.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='output_files'},indent=2))
