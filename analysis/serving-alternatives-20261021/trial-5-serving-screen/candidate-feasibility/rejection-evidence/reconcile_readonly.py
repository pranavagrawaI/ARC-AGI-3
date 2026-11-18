import datetime as dt,json
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
p=Path(__file__).resolve().parent
api=KaggleApi();api.authenticate()
out={'utc':dt.datetime.now(dt.timezone.utc).isoformat()}
for key,search in [('exact_search','serving-candidate-feas5-20261023'),('no_match_control','no-such-kernel-feas5-reconciliation-20261023')]:
    r=api.kernels_list_with_response(mine=True,user='pranavagrawai',sort_by='dateRun',page_size=20,search=search)
    out[key]=type(r).to_dict(r)
try:
    r=api.kernels_status('pranavagrawai/serving-candidate-feas5-20261023');out['status']=type(r).to_dict(r)
except Exception as e:
    r=getattr(e,'response',None)
    out['status']={'error_type':type(e).__name__,'status':None if r is None else r.status_code,'body':None if r is None else r.text[:8192]}
with api.build_kaggle_client() as c:q=c.kernels.kernels_api_client.get_accelerator_quota_statistics()
out['quota']={'utc':dt.datetime.now(dt.timezone.utc).isoformat(),'used_s':q.gpu_quota.time_used.total_seconds(),'reserved_s':q.gpu_quota.time_reserved.total_seconds(),'delta_used_s':q.gpu_quota.time_used.total_seconds()-59265.958}
(p/'reconciliation.json').write_text(json.dumps(out,indent=2,default=str)+'\n')
print(json.dumps(out,indent=2,default=str))
