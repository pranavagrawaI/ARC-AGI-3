"""Unsealed watcher: wait until no kernel is active, then run the sealed launcher. execute.py repeats every admission
check and its exclusive attempt receipt makes a second push impossible; pre-push refusals consume nothing."""
import json,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from launch import now
from kaggle_backend import KaggleBackend

log=HERE/'evidence/wait-and-launch.jsonl'
def note(**row):
    with log.open('a') as s:s.write(json.dumps({'utc':now(),**row})+'\n')

deadline=time.time()+12*3600
while time.time()<deadline:
    if (HERE/'evidence/launch-attempt.json').exists():note(state='attempt exists; watcher done');break
    if (HERE/'readout/collector-ready.json').exists():note(state='collector identity exists; needs reconciliation; watcher stops');break
    try:active=KaggleBackend().active_sessions('pranavagrawai')
    except Exception as e:note(state='poll error',error=type(e).__name__);time.sleep(120);continue
    if active:note(state='waiting',active=active);time.sleep(120);continue
    done=subprocess.run([sys.executable,'-B',str(HERE/'execute.py')],cwd=HERE,capture_output=True,text=True)
    note(state='execute ran',returncode=done.returncode,stdout=done.stdout[-1500:])
    if done.returncode==0 or (HERE/'evidence/launch-attempt.json').exists():break
    time.sleep(120)
else:note(state='12 h deadline; no launch')
