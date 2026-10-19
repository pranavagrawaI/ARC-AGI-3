"""Bounded read-only sample of the installed SDK's live session log stream."""
import datetime as dt
import json
from pathlib import Path
import signal
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]/'analysis/marlin-fixed-replay-r3-20261011'))
from kaggle_backend import KaggleBackend
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionLogsStreamRequest


def main():
    backend=KaggleBackend()
    accepted=json.loads((HERE/'launch-response.json').read_text())
    slug=json.loads((HERE/'kernel/kernel-metadata.json').read_text())['id']
    meta=backend.get_kernel(slug).metadata
    assert meta.id==accepted['kernelId'] and meta.current_version_number==accepted['versionNumber']
    request=ApiGetKernelSessionLogsStreamRequest()
    request.user_name,request.kernel_slug=slug.split('/')
    request.version_label=str(accepted['versionNumber'])
    request.wait_for_logs_url_seconds=5
    lines=[];response=None;kind=None;error=None
    def timeout(*args):
        raise TimeoutError('bounded live-log sampling deadline')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(25)
    started=time.monotonic()
    try:
        with backend.api.build_kaggle_client() as client:
            response=client.kernels.kernels_api_client.get_kernel_session_logs_stream(request)
            kind=response.headers.get('Content-Type')
            for line in response.iter_lines(chunk_size=1024,decode_unicode=True):
                if line:
                    lines.append(line if isinstance(line,str) else line.decode(errors='replace'))
                if time.monotonic()-started>15 or len(lines)>100000:
                    break
    except Exception as exc:
        error=dict(type=type(exc).__name__,status=getattr(getattr(exc,'response',None),'status_code',None))
    finally:
        signal.alarm(0)
        if response is not None:response.close()
        raw='\n'.join(lines)+'\n'
        (HERE/'live-session-log.raw.txt').write_text(raw)
        row=dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),kernel_id=meta.id,
                 version=meta.current_version_number,content_type=kind,lines=len(lines),
                 elapsed_s=round(time.monotonic()-started,2),error=error)
        (HERE/'live-log-receipt.json').write_text(json.dumps(row,indent=2)+'\n')
        print(json.dumps(row))
        print('\n'.join(lines[-8:]))


if __name__=='__main__':main()
