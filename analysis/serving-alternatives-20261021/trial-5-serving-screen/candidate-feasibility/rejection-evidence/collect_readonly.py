import datetime as dt, hashlib, json
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
ROOT=Path(__file__).resolve().parent
REF='pranavagrawai/serving-candidate-feas5-20261023'
def now(): return dt.datetime.now(dt.timezone.utc).isoformat()
def save(name,data): (ROOT/name).write_text(json.dumps(data,indent=2,default=str)+'\n')
def serial(o): return type(o).to_dict(o)
def error(e):
    out={'type':type(e).__name__,'message':str(e)}
    r=getattr(e,'response',None)
    if r is not None:
        out.update(status=r.status_code,headers={k:v for k,v in r.headers.items() if k.lower() in ('content-type','date','content-length','x-request-id','x-kaggle-request-id','retry-after')},body=r.text[:8192],body_truncated=len(r.content)>8192)
    return out
api=KaggleApi();api.authenticate()
report={'started_utc':now(),'ref':REF,'operation_policy':'read-only; no SaveKernel, push, cancel or retry'}
try:
    report['exact_status']={'utc':now(),'response':serial(api.kernels_status(REF))}
except Exception as e: report['exact_status']={'utc':now(),'error':error(e)}
save('readback.json',report)
try:
    with api.build_kaggle_client() as c:
        req=ApiGetKernelRequest();req.user_name='pranavagrawai';req.kernel_slug=REF.split('/')[1]
        got=c.kernels.kernels_api_client.get_kernel(req)
    save('exact-kernel-readback.json',serial(got))
    report['exact_get']={'utc':now(),'result':'found','artifact':'exact-kernel-readback.json'}
except Exception as e: report['exact_get']={'utc':now(),'error':error(e)}
save('readback.json',report)
try:
    listing=api.kernels_list(mine=True,user='pranavagrawai',sort_by='dateRun',page_size=20)
    sessions=[]
    for k in listing:
        row={'ref':k.ref}
        try:
            result=api.kernels_status(k.ref);state=result.status
            row['status']=getattr(state,'name',str(state)).upper()
        except Exception as e:row['error']=error(e)
        sessions.append(row)
    report['sessions']={'utc':now(),'entries':sessions,'candidate_in_recent_list':any(x['ref']==REF for x in sessions)}
except Exception as e:report['sessions']={'utc':now(),'error':error(e)}
save('readback.json',report)
try:
    with api.build_kaggle_client() as c:q=c.kernels.kernels_api_client.get_accelerator_quota_statistics()
    report['quota']={'utc':now(),'used_s':q.gpu_quota.time_used.total_seconds(),'reserved_s':q.gpu_quota.time_reserved.total_seconds(),'delta_from_prelaunch_s':q.gpu_quota.time_used.total_seconds()-59265.958}
except Exception as e:report['quota']={'utc':now(),'error':error(e)}
report['finished_utc']=now()
save('readback.json',report)
print(json.dumps(report,indent=2))
