"""Read-only Kaggle monitor; never launches, retries, or submits a run."""
from __future__ import annotations
import argparse
import datetime as dt
import fcntl
import json
import subprocess
import time
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kernel')
    parser.add_argument('directory',type=Path)
    parser.add_argument('--hours',type=float,default=4)
    parser.add_argument('--version',type=int,required=True)
    parser.add_argument('--legacy-index',type=Path)
    parser.add_argument('--bulk',action='store_true')
    args=parser.parse_args()
    args.directory.mkdir(parents=True,exist_ok=True)
    with (args.directory/'collector.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        until=time.monotonic()+args.hours*3600
        while time.monotonic()<until:
            try:
                result=subprocess.run(['kaggle','kernels','status',args.kernel],
                    capture_output=True,text=True,timeout=60)
                text=result.stdout+result.stderr
                record={'utc':dt.datetime.now(dt.timezone.utc).isoformat(),
                    'returncode':result.returncode,'status':text.strip()}
            except subprocess.TimeoutExpired:
                record={'utc':dt.datetime.now(dt.timezone.utc).isoformat(),
                    'returncode':None,'status':'status lookup timed out; remote state unknown'}
                text=''
            with (args.directory/'status-history.jsonl').open('a') as stream:
                stream.write(json.dumps(record)+'\n')
            (args.directory/'status.txt').write_text(record['status']+'\n')
            terminal=any('KernelWorkerStatus.'+state in text for state in
                ('COMPLETE','ERROR','CANCELLED','CANCELED','FAILED'))
            if record['returncode']==0 and terminal:
                with (args.directory/'download.log').open('w') as log:
                    try:
                        import sys
                        command=[sys.executable,str(Path(__file__).with_name('collect_outputs.py')),
                                 args.kernel,str(args.directory),'--version',str(args.version)]
                        if args.legacy_index: command+=['--legacy-index',str(args.legacy_index)]
                        if args.bulk: command+=['--bulk']
                        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=1200)
                        receipt_path=args.directory/'collection-receipt.json'
                        receipt=json.loads(receipt_path.read_text()) if receipt_path.exists() else {
                            'critical_complete':False,'error':'collector exited before producing a receipt'}
                        receipt['download_returncode']=result.returncode
                    except subprocess.TimeoutExpired:
                        receipt_path=args.directory/'collection-receipt.json'
                        receipt=json.loads(receipt_path.read_text()) if receipt_path.exists() else {'critical_complete':False}
                        receipt.update(download_returncode=None,error='download timeout; preserve partial files')
                receipt.update(remote_status=record,interpretation='Artifacts require inspection; no efficacy verdict.')
                (args.directory/'collection-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
                # Preserve a deterministic receipt audit even when the interactive
                # agent is no longer attached. It cannot launch follow-up work.
                subprocess.run([sys.executable,str(Path(__file__).with_name('audit_qualification.py')),
                                str(args.directory)],capture_output=True,text=True,timeout=60)
                return
            time.sleep(30)
        (args.directory/'collection-receipt.json').write_text(json.dumps({
            'error':'monitor window ended; remote state remains unresolved; do not relaunch'},indent=2)+'\n')

if __name__=='__main__':
    main()
