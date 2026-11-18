import json,datetime
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
out=Path(__file__).resolve().parent;api=KaggleApi();api.authenticate();r=ApiListKernelSessionOutputRequest();r.user_name='pranavagrawai';r.kernel_slug='r10a-cd82-constructor-repair';r.page_size=200
rec={'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'CURRENT supplemental filenames ONLY; no downloads or version proof','request':r.to_dict()}
try:
 with api.build_kaggle_client() as c:res=c.kernels.kernels_api_client.list_kernel_session_output(r)
 raw=res.to_dict();raw.pop('log',None)
 for row in raw.get('files',[]):
  if 'url' in row:row['url']='[REDACTED signed download URL]'
 rec['response']=raw;print(json.dumps(raw,indent=2)[:3500])
except Exception as e:rec['error']={'type':type(e).__name__,'message':str(e)};print(rec['error'])
with (out/'supplemental-current-output-discovery.json').open('x') as f:json.dump(rec,f,indent=2)
