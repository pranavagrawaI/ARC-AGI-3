import sys,json,hashlib,datetime,traceback
from pathlib import Path
root=Path('/home/pranav/projects/arc-agi-3/analysis/serving-alternatives-20261021')
p=root/'trial-17-candidate-first-pair-r2-local';out=root/'coordination-20261024/trial17-r2-root-preflight'
out.mkdir(exist_ok=False)
sys.path.insert(0,str(p))
import launch
from kaggle.api.kaggle_api_extended import KaggleApi
launch.HERE=out
api=KaggleApi();api.authenticate()
def forbidden(*a,**kw):raise RuntimeError('Root preflight forbids SaveKernel')
api.kernels_push=forbidden
try:
 evidence=launch.fresh_checks(api)
 result={'status':'READ_ONLY_PREFLIGHT_PASS','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'evidence':evidence,'SaveKernel_calls':0,'launch_attempt':False}
except BaseException as e:
 result={'status':'READ_ONLY_PREFLIGHT_FAILED','type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc(),'SaveKernel_calls':0,'launch_attempt':False}
(out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'status':result['status'],'path':str(out/'RESULT.json'),'sha256':hashlib.sha256((out/'RESULT.json').read_bytes()).hexdigest()},indent=2))
