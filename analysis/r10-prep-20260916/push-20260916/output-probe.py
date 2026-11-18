import json,datetime,uuid
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
out=Path(__file__).resolve().parent;api=KaggleApi();api.authenticate()
r=ApiListKernelSessionOutputRequest();r.user_name='pranavagrawai';r.kernel_slug='r10a-cd82-constructor-repair';r.version_label='1';r.page_size=200
rec={'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'EXPLICIT_VERSION_RECOVERY_OUTPUT_LIST','request':r.to_dict()}
try:
 with api.build_kaggle_client() as c:res=c.kernels.kernels_api_client.list_kernel_session_output(r)
 raw=res.to_dict()
 # Signed download URLs are capabilities, retain nonsecret file metadata only.
 for row in raw.get('files',[]):
  if 'url' in row:row['url']='[REDACTED signed download URL]'
 rec['response']=raw
 print(json.dumps({k:v for k,v in raw.items() if k!='log'},indent=2))
except Exception as e:rec['error']={'type':type(e).__name__,'message':str(e)};print(rec['error'])
with (out/('output-list-'+uuid.uuid4().hex+'.json')).open('x') as f:json.dump(rec,f,indent=2,default=str)
