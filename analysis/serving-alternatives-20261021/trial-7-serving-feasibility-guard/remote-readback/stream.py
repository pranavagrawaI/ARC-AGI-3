"""Read-only SDK execution-log stream. No remote execution/mutation."""
import datetime as dt,json,sys
from pathlib import Path
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent))
from http_error_capture import exception_record
from kaggle.api.kaggle_api_extended import KaggleApi
api=KaggleApi();api.authenticate()
with (P/'live-events.jsonl').open('x') as output:
    try:
        for event in api.kernels_logs_stream('pranavagrawai/serving-candidate-feas7-20261023'):
            output.write(json.dumps(dict(received_utc=dt.datetime.now(dt.timezone.utc).isoformat(),event=event))+'\n');output.flush()
    except Exception as error:
        (P/'live-stream-error.json').write_text(json.dumps(exception_record(error),indent=2)+'\n')
        raise
