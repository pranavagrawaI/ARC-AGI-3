import json,datetime,uuid
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest,ApiGetKernelSessionStatusRequest
out=Path(__file__).resolve().parent;api=KaggleApi();api.authenticate()
for suffix,version in [('', '1'),('/1','1'),('',None)]:
 for cls,method in [(ApiGetKernelRequest,'get_kernel'),(ApiGetKernelSessionStatusRequest,'get_kernel_session_status')]:
  r=cls();r.user_name='pranavagrawai';r.kernel_slug='r10a-cd82-constructor-repair'+suffix
  if version:r.version_label=version
  receipt={'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'RECOVERY ONLY; latest diagnostic is not exact-version proof','method':method,'request':r.to_dict()}
  try:
   with api.build_kaggle_client() as c:res=getattr(c.kernels.kernels_api_client,method)(r)
   receipt['response']=res.to_dict();print(method,suffix,version,json.dumps(res.metadata.to_dict() if method=='get_kernel' else res.to_dict(),default=str))
  except Exception as e:receipt['error']={'type':type(e).__name__,'message':str(e)};print(method,suffix,version,receipt['error'])
  with (out/('recovery-'+uuid.uuid4().hex+'.json')).open('x') as f:json.dump(receipt,f,indent=2,default=str)
