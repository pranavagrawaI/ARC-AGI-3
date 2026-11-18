import json,datetime,hashlib
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest,ApiGetKernelSessionStatusRequest
out=Path(__file__).resolve().parent
api=KaggleApi();api.authenticate()
for slug,label in [('r10a-cd82-constructor-repair','returned'),('arc3-r10a-cd82-20260916','intended')]:
 r=ApiGetKernelRequest();r.user_name='pranavagrawai';r.kernel_slug=slug;r.version_label='1'
 receipt={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'request':r.to_dict()}
 try:
  with api.build_kaggle_client() as c:res=c.kernels.kernels_api_client.get_kernel(r)
  receipt['response']=res.to_dict();print(label,json.dumps(res.metadata.to_dict()))
 except Exception as e:receipt['error']={'type':type(e).__name__,'message':str(e)};print(label,receipt['error'])
 with (out/(label+'-readback.json')).open('x') as f:json.dump(receipt,f,indent=2,default=str)
 r=ApiGetKernelSessionStatusRequest();r.user_name='pranavagrawai';r.kernel_slug=slug;r.version_label='1'
 receipt={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'request':r.to_dict()}
 try:
  with api.build_kaggle_client() as c:res=c.kernels.kernels_api_client.get_kernel_session_status(r)
  receipt['response']=res.to_dict();print(label,'status',res.to_dict())
 except Exception as e:receipt['error']={'type':type(e).__name__,'message':str(e)};print(label,'status',receipt['error'])
 with (out/(label+'-status.json')).open('x') as f:json.dump(receipt,f,indent=2,default=str)
