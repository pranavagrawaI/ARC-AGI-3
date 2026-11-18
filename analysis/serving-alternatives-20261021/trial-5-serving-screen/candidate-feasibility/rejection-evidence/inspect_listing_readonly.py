import datetime as dt,json
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
p=Path(__file__).resolve().parent
api=KaggleApi();api.authenticate()
result=api.kernels_list_with_response(mine=True,user='pranavagrawai',sort_by='dateRun',page_size=20)
data=type(result).to_dict(result)
(p/'listing-readback.json').write_text(json.dumps({'utc':dt.datetime.now(dt.timezone.utc).isoformat(),'response':data},indent=2,default=str)+'\n')
for k in api.kernels_list(mine=True,user='pranavagrawai',sort_by='dateRun',page_size=20):
    if not k.ref: print(json.dumps(type(k).to_dict(k),indent=2,default=str))
