import json,datetime,time,uuid
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionStatusRequest
out=Path(__file__).resolve().parent/'monitor';out.mkdir(exist_ok=False)
api=KaggleApi();api.authenticate();deadline=time.monotonic()+3900
while time.monotonic()<deadline:
 terminal=False
 for version in ['1',None]:
  r=ApiGetKernelSessionStatusRequest();r.user_name='pranavagrawai';r.kernel_slug='r10a-cd82-constructor-repair'
  if version:r.version_label=version
  rec={'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'EXPLICIT_VERSION_RECOVERY' if version else 'CURRENT_SUPPLEMENTAL_NOT_VERSION_PROOF','request':r.to_dict()}
  try:
   with api.build_kaggle_client() as c:res=c.kernels.kernels_api_client.get_kernel_session_status(r)
   rec['response']=res.to_dict();terminal=res.status.name in ['COMPLETE','ERROR','CANCEL_ACKNOWLEDGED']
  except Exception as e:rec['error']={'type':type(e).__name__,'message':str(e)}
  path=out/(uuid.uuid4().hex+'.json')
  with path.open('x') as f:json.dump(rec,f,indent=2)
  print(json.dumps({'receipt':str(path),'scope':rec['scope'],'result':rec.get('response',rec.get('error'))}),flush=True)
  if terminal:
   with (out/'TERMINAL-OBSERVATION.json').open('x') as f:json.dump({'scope':rec['scope'],'receipt':str(path),'exact_version_established':version is not None,'response':rec['response']},f,indent=2)
   break
 if terminal:break
 time.sleep(45)
