"""Sequential isolated continuations with a total deadline; no server launch."""
import argparse
import hashlib
import json
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
    for row in matrix:
        if any(hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected for path,expected in frozen.items()):
            raise RuntimeError('Frozen source/input changed mid-matrix')
        if time.monotonic()>=deadline:
            raise TimeoutError('Matrix deadline')
        command=[sys.executable,'-B',str(source),'--checkpoint',row['checkpoint'],'--variant',row['variant'],
                 '--seed',str(row['seed']),'--mode',args.mode,'--output',str(output/row['id']),'--url',args.url]
        with (output/(row['id']+'.log')).open('x') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,
                           timeout=min(960,deadline-time.monotonic()),check=True)
        result=json.loads((output/row['id']/'result.json').read_text())
        if not result.get('qualified_restoration') or not result.get('protected_inputs_unchanged') or result.get('unresolved'):
            raise RuntimeError('Invalid episode; stopping without retry: '+row['id'])
        if args.mode=='live' and result['status'] not in ('solved','token_budget','request_budget','episode_deadline','gave_up'):
            raise RuntimeError('Failed live episode; stopping without retry: '+row['id'])
        print(row['id'],result['status'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--mode',choices=['restore','live'],default='restore')
    parser.add_argument('--url',default='http://127.0.0.1:8001/v1')
    main(parser.parse_args())
