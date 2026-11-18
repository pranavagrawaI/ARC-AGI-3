import datetime as dt,json
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
P=Path(__file__).resolve().parent
api=KaggleApi();api.authenticate()
ref='pranavagrawai/serving-candidate-feas6-20261023'
with api.build_kaggle_client() as c:
    request=ApiGetKernelRequest();request.user_name='pranavagrawai';request.kernel_slug='serving-candidate-feas6-20261023/1'
    response=c.kernels.kernels_api_client.get_kernel(request)
(P/'get-kernel-v1.json').write_text(json.dumps(type(response).to_dict(response),indent=2)+'\n')
(P/'source-v1.ipynb').write_text(response.blob.source)
(P/'metadata-v1.json').write_text(json.dumps(type(response.metadata).to_dict(response.metadata),indent=2)+'\n')
state=api.kernels_status(ref)
with api.build_kaggle_client() as c:q=c.kernels.kernels_api_client.get_accelerator_quota_statistics()
status=dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),kernel_ref=ref,version=1,kernel_id=140341756,
            status=type(state).to_dict(state),quota={'used_s':q.gpu_quota.time_used.total_seconds(),'reserved_s':q.gpu_quota.time_reserved.total_seconds()})
(P/'initial-status-quota.json').write_text(json.dumps(status,indent=2)+'\n')
print(json.dumps(status,indent=2))
