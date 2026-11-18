"""Read-only small primary receipts while full version-bound archive downloads."""
import datetime as dt,json,sys
from pathlib import Path
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P.parent))
from http_error_capture import exception_record
from kaggle.api.kaggle_api_extended import KaggleApi
api=KaggleApi();api.authenticate()
destination=P/'priority-output';destination.mkdir(exist_ok=False)
pattern=r'/output/(result\.json|supervisor\.jsonl|hardware\.(stdout|stderr)|control\.(server|prepare|identity|replay)\.(stdout|stderr))$|/control-package/(actual-gpu-query|actual-nvidia-smi|host-headroom|control-prepared|control-health-identity|control-health-serve)\.json$'
report={'started_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'kernel_ref':'pranavagrawai/serving-wire-capacity9-20261023','version':1,'kernel_id':140375737,'purpose':'small primary receipts; full version-bound archive collected separately; exclusive new destination'}
try:
 files,token=api.kernels_output(report['kernel_ref'],path=str(destination),file_pattern=pattern,force=False,quiet=True,page_size=100)
 report.update(files=files,next_page_token=token,completed_utc=dt.datetime.now(dt.timezone.utc).isoformat())
except Exception as e:report['error']=exception_record(e)
(P/'priority-collection.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
