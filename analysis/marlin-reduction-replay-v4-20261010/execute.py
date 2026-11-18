"""Verify/admit, arm collector, recheck admission, push once. Never retry a push."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from launch import HERE,now,preflight,push_once,save,sha,verify_local


def start_collector(root=HERE):
    root=Path(root);readout=root/'readout';readout.mkdir(exist_ok=True)
    if (readout/'collector-ready.json').exists():raise ValueError('Existing collector identity requires reconciliation')
    with (readout/'collector.log').open('xb') as log:
        proc=subprocess.Popen([sys.executable,'-B',str(root/'collect.py')],cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    until=time.monotonic()+20
    while time.monotonic()<until:
        if proc.poll() is not None:raise RuntimeError('Collector exited before readiness')
        ready=readout/'collector-ready.json'
        if ready.exists():
            receipt=json.loads(ready.read_text())
            if receipt['pid']!=proc.pid or receipt['freeze_sha256']!=sha(root/'launch-freeze.json') or not receipt['read_only']:
                raise ValueError('Collector ready receipt differs from the live child')
            return proc
        time.sleep(.1)
    raise TimeoutError('Collector did not become ready within20 seconds')


def execute(backend,root=HERE,collector=start_collector,dry=False):
    root=Path(root);evidence=root/'evidence';evidence.mkdir(exist_ok=True)
    if (evidence/'launch-attempt.json').exists():raise ValueError('Launch attempt already exists; no retry')
    proc=None
    try:
        admission=preflight(backend,root)
        if dry:return {'state':'admitted; dry run only','admission':admission}
        # Preparation authority ("get ready for launch") is not a push; the user's go is recorded separately.
        if not (root/'evidence/GO.json').exists():raise PermissionError('No recorded go instruction; dry run only')
        proc=collector(root)
        # Collector startup precedes the irreversible boundary; check live admission again.
        admission=preflight(backend,root)
        if proc.poll() is not None:raise ValueError('Collector stopped before the push boundary')
        response=push_once(backend,admission,root)
        return {'state':'accepted','response':response,'collector_pid':proc.pid}
    except Exception as error:
        if proc is not None and not (evidence/'launch-attempt.json').exists():
            if proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill();proc.wait(timeout=5)
            save(root/'readout/collector-stop.json',{'utc':now(),'reason':'pre-push failure; owned collector stopped; no GPU push'})
        # Only exception types are persisted; SDK exceptions may contain signed URLs.
        path=evidence/'preflight-history.jsonl'
        with path.open('a') as stream:stream.write(json.dumps({'utc':now(),'state':'launch incomplete','error_type':type(error).__name__,'push_attempt_recorded':(evidence/'launch-attempt.json').exists()})+'\n')
        raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dry',action='store_true');args=parser.parse_args()
    try:
        verify_local()
        from kaggle_backend import KaggleBackend
        backend=KaggleBackend()
        result=execute(backend,dry=args.dry);save(HERE/'evidence/execute-result.json',result);print(json.dumps(result,indent=2))
    except Exception as error:
        result={'utc':now(),'state':'not launched' if not (HERE/'evidence/launch-attempt.json').exists() else 'inspect exclusive attempt; do not retry',
                'error_type':type(error).__name__,'authority':'single replay launch authorized by user; no repeated approval needed',
                'launch_attempt_recorded':(HERE/'evidence/launch-attempt.json').exists()}
        save(HERE/'evidence/execute-result.json',result);print(json.dumps(result,indent=2));raise SystemExit(1)


if __name__=='__main__':main()
