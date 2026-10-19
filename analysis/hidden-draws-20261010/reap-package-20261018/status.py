"""One read-only status/log refresh for the accepted kernel; never launches."""
import datetime as dt
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]/'analysis/marlin-fixed-replay-r3-20261011'))
from kaggle_backend import KaggleBackend


def main():
    response=json.loads((HERE/'launch-response.json').read_text())
    slug=json.loads((HERE/'kernel/kernel-metadata.json').read_text())['id']
    b=KaggleBackend()
    meta=b.get_kernel(slug).metadata
    assert meta.id==response['kernelId'] and meta.current_version_number==response['versionNumber']
    status=b.status(slug)
    files,token,raw=b.output_page(slug,None)
    log=''
    if raw:
        (HERE/'latest-kernel-log.raw.json').write_text(raw)
        try:
            rows=json.loads(raw)
            log=''.join(row.get('data','') for row in rows)
        except (TypeError,json.JSONDecodeError):
            log=raw
        (HERE/'latest-kernel.log').write_text(log)
    row={'utc':dt.datetime.now(dt.timezone.utc).isoformat(),**status,
         'kernel_id':meta.id,'version':meta.current_version_number,
         'files_first_page':[f.file_name for f in files],'more_output_pages':bool(token)}
    with (HERE/'status-history.jsonl').open('a') as stream:
        stream.write(json.dumps(row)+'\n')
    print(json.dumps({k:v for k,v in row.items() if k!='files_first_page'}))
    print('\n'.join(log.splitlines()[-8:]))


if __name__=='__main__':
    main()
