"""One filtered read-only fetch of only required files absent from partial ZIP."""
import datetime as dt,json,sys
from pathlib import Path
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P.parent))
from http_error_capture import exception_record
from kaggle.api.kaggle_api_extended import KaggleApi
api=KaggleApi();api.authenticate();destination=P/'missing-raw';destination.mkdir(exist_ok=False)
r=dict(started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),kernel_ref='pranavagrawai/serving-wire-capacity10-20261023',kernel_id=140391414,version=1,scope='ONLY ABSENT RAW CONTROL FILES; NO ARCHIVE RETRY',requested=json.loads((P/'raw-missing-files.json').read_text()))
try:
 files,token=api.kernels_output(r['kernel_ref'],path=str(destination),file_pattern=(P/'missing-file-pattern.txt').read_text(),force=False,quiet=True,page_size=100)
 r.update(files=files,next_page_token=token,completed_utc=dt.datetime.now(dt.timezone.utc).isoformat())
except Exception as e:r['error']=exception_record(e)
(P/'missing-raw-collection.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k not in ('files','requested')},indent=2))
