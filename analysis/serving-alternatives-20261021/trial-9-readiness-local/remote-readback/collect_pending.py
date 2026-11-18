"""Read-only exact ten pending receipts and request550 capture; no duplicate archive."""
import datetime as dt,json,re,sys
from pathlib import Path
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P.parent))
from kaggle.api.kaggle_api_extended import KaggleApi
from http_error_capture import exception_record
rows=[json.loads(x) for x in (P/'priority-events/serving-wire-capacity9/output/control/events.jsonl').read_text().splitlines()]
rids=[x['rid'] for x in rows if x['kind']=='cutoff_pending'];assert len(rids)==10
rid='serving-wire-capacity9-20261023-control-550';assert rid in rids
pattern=r'/('+ '|'.join(re.escape(x) for x in rids)+r')/http\.json$|/'+re.escape(rid)+r'/(wire\.http|wire\.json|body\.bin)$'
destination=P/'priority-pending';destination.mkdir(exist_ok=False);api=KaggleApi();api.authenticate()
report={'started_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'kernel_ref':'pranavagrawai/serving-wire-capacity9-20261023','version':1,'kernel_id':140375737,'requested_pending_ids':rids,'requested_failed_capture_files':['http.json','wire.http','wire.json','body.bin'],'purpose':'exact pending clocks/transport; full archive already active'}
try:
 files,token=api.kernels_output(report['kernel_ref'],path=str(destination),file_pattern=pattern,force=False,quiet=True,page_size=100)
 report.update(files=files,next_page_token=token,completed_utc=dt.datetime.now(dt.timezone.utc).isoformat())
except Exception as e:report['error']=exception_record(e)
(P/'pending-collection.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
