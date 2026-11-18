"""Actual malformed HTTP parse + scheduling injection; CPU, not incidence evidence."""
import argparse
import hashlib
import http.client
import json
from pathlib import Path
import shutil
import sys
import time
from unittest.mock import patch

def main():
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('output');a=p.parse_args()
    source=Path(a.source).resolve();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(source));sys.path.insert(0,str(source/'tests'))
    import test_replay,replay,reduce_api
    result={'scope':'CPU_ACTUAL_HTTP_PARSER_WITH_EXPLICIT_RETHROW_DELAY_NOT_INCIDENCE_CLAIM',
            'source':str(source),'source_sha256':{n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in ('replay.py','reduce_api.py')}}
    for mode in ('malformed','remote_eof'):
        detected=[];actual=http.client.HTTPResponse._read_status
        def serve(handler):
            body=handler.rfile.read(int(handler.headers['Content-Length']))
            handler.server.seen.append((handler.headers['X-Request-Id'],body,time.monotonic()))
            if mode=='malformed':handler.wfile.write(b'HTTP/1.1 200 '+b'X'*65536+b'\r\n\r\n');handler.wfile.flush()
            else:time.sleep(.3)
        def delayed(response):
            try:return actual(response)
            except http.client.LineTooLong:
                detected.append(time.monotonic());time.sleep(.3);raise
        case=test_replay.ReplayTests()
        try:
            with patch.object(test_replay.Handler,'do_POST',serve),patch.object(http.client.HTTPResponse,'_read_status',delayed):
                root,server,obj,events,ok,elapsed=case.run_fixture('ok',(.05,.05))
            shutil.copytree(root/'out',out/mode/'raw')
            receipts=[json.loads(q.read_text()) for q in (root/'out').glob('*/http.json')];accepted=[];issues=[]
            for receipt in receipts:
                try:reduce_api.pending_http(receipt,obj.end,lambda ref:(root/'out'/ref).read_bytes(),receipt['rid'])
                except (ValueError,KeyError) as error:accepted.append(False);issues.append(str(error))
                else:accepted.append(True);issues.append(None)
            result[mode]={'run_ok':ok,'failed':obj.failed,'parser_detected_at':detected,'parser_error_before_cutoff':[t<obj.end for t in detected],
                          'replay_end':obj.end,'pending_accepted':accepted,'pending_issues':issues,'elapsed_s':elapsed,'receipts':receipts}
        finally:case.doCleanups()
    (out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    inventory={str(q.relative_to(out)):hashlib.sha256(q.read_bytes()).hexdigest() for q in out.rglob('*') if q.is_file()}
    (out/'INVENTORY.json').write_text(json.dumps(inventory,indent=2)+'\n')
    print(json.dumps({mode:{k:v for k,v in result[mode].items() if k not in ('receipts','parser_detected_at')} for mode in ('malformed','remote_eof')},indent=2))

if __name__=='__main__':main()
