"""Sequential isolated continuations with a total deadline; no server launch."""
import argparse
import hashlib
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time
from continuation import identity_files

HERE=Path(__file__).resolve().parent


def main(args):
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    matrix=json.loads((HERE/'matrix.json').read_text())
    (output/'matrix.json').write_text(json.dumps(matrix,indent=2)+'\n')
    source=HERE/'continuation.py'
    frozen={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in identity_files(['b1','control'])}
    frozen[str(HERE/'matrix.json')]=hashlib.sha256((HERE/'matrix.json').read_bytes()).hexdigest()
    (output/'matrix-identity.json').write_text(json.dumps({'source_sha256':frozen,'mode':args.mode,
        'episode_tokens':32768,'total_tokens':len(matrix)*32768,'total_seconds':14400},indent=2)+'\n')
    deadline=time.monotonic()+14400
    session_deadline=float(os.environ.get('PLANNING_MATRIX_DEADLINE_EPOCH',time.time()+14400))
    results=[]
    for row in matrix:
        if any(hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected for path,expected in frozen.items()):
            raise RuntimeError('Frozen source/input changed mid-matrix')
        remaining=min(deadline-time.monotonic(),session_deadline-time.time())
        if remaining<=0:
            raise TimeoutError('Matrix deadline')
        command=[sys.executable,'-B',str(source),'--checkpoint',row['checkpoint'],'--variant',row['variant'],
                 '--seed',str(row['seed']),'--mode',args.mode,'--output',str(output/row['id']),'--url',args.url]
        with (output/(row['id']+'.log')).open('x') as log:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:status=process.wait(timeout=min(960,remaining))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait();status=-9
        result_path=output/row['id']/'result.json'
        result=json.loads(result_path.read_text()) if result_path.exists() else {'status':'missing_result'}
        results.append({**row,**result,'exitcode':status})
        (output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
        if status:
            raise RuntimeError('Worker failed; no retry: '+row['id'])
        if not result.get('qualified_restoration') or not result.get('protected_inputs_unchanged') or result.get('unresolved'):
            raise RuntimeError('Invalid episode; stopping without retry: '+row['id'])
        if args.mode=='live' and result['status'] not in ('solved','token_budget','request_budget','action_budget','episode_deadline','gave_up'):
            raise RuntimeError('Failed live episode; stopping without retry: '+row['id'])
        print(row['id'],result['status'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--mode',choices=['restore','live'],default='restore')
    parser.add_argument('--url',default='http://127.0.0.1:8001/v1')
    main(parser.parse_args())
