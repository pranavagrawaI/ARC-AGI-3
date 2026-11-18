import json,datetime,uuid
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest,ApiGetKernelSessionStatusRequest
out=Path(__file__).resolve().parent;api=KaggleApi();api.authenticate()
for version in ['1',None]:
 for cls,method in [(ApiGetKernelRequest,'get_kernel'),(ApiGetKernelSessionStatusRequest,'get_kernel_session_status')]:
  r=cls();r.user_name='pranavagrawai';r.kernel_slug='r10a-cd82-constructor-repair'
  if version:r.version_label=version
  rec={'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'EXPLICIT_VERSION_RECOVERY' if version else 'CURRENT_SUPPLEMENTAL_NOT_VERSION_PROOF','method':method,'request':r.to_dict()}
  try:
   with api.build_kaggle_client() as c:res=getattr(c.kernels.kernels_api_client,method)(r)
   rec['response']=res.to_dict();print(version,method,json.dumps(res.metadata.to_dict() if method=='get_kernel' else res.to_dict(),default=str))
  except Exception as e:rec['error']={'type':type(e).__name__,'message':str(e)};print(version,method,rec['error'])
  with (out/('terminal-readback-'+uuid.uuid4().hex+'.json')).open('x') as f:json.dump(rec,f,indent=2,default=str)
r={'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'method':'quota_view'}
try:r['response']=api.quota_view().to_dict()
except Exception as e:r['error']={'type':type(e).__name__,'message':str(e)}
with (out/'terminal-quota.json').open('x') as f:json.dump(r,f,indent=2,default=str)
print('quota',json.dumps(r,default=str))
