import json,datetime,uuid
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
out=Path(__file__).resolve().parent;api=KaggleApi();api.authenticate();original=api.get_response_processor()
def capture(response):
 with (out/'supplemental-current-raw-response.json').open('x') as f:f.write(response.text)
 if original:original(response)
api.get_response_processor=lambda:capture
r=ApiGetKernelRequest();r.user_name='pranavagrawai';r.kernel_slug='r10a-cd82-constructor-repair'
with (out/'supplemental-current-raw-request.json').open('x') as f:json.dump({'scope':'LATEST supplemental only','time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'request':r.to_dict()},f)
with api.build_kaggle_client() as c:res=c.kernels.kernels_api_client.get_kernel(r)
raw=json.loads((out/'supplemental-current-raw-response.json').read_text());print(json.dumps({k:v for k,v in raw.items() if k!='blob'},indent=2));print('blob fields',list(raw.get('blob',{})))
