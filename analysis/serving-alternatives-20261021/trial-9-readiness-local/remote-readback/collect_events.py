"""Read-only single replay event ledger while the one full archive downloads."""
import datetime as dt,json,sys
from pathlib import Path
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P.parent))
from http_error_capture import exception_record
from kaggle.api.kaggle_api_extended import KaggleApi
api=KaggleApi();api.authenticate();destination=P/'priority-events';destination.mkdir(exist_ok=False)
r={'started_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'kernel_ref':'pranavagrawai/serving-wire-capacity9-20261023','version':1,'kernel_id':140375737,'purpose':'only control events; no duplicate archive'}
try:
 files,token=api.kernels_output(r['kernel_ref'],path=str(destination),file_pattern=r'/output/control/events\.jsonl$',force=False,quiet=True,page_size=100)
 r.update(files=files,next_page_token=token,completed_utc=dt.datetime.now(dt.timezone.utc).isoformat())
except Exception as e:r['error']=exception_record(e)
(P/'events-collection.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r,indent=2))
