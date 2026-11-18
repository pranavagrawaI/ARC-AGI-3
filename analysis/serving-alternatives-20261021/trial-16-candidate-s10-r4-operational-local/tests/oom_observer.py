"""Explicit CPU input injection only; actual observer lifecycle, no cgroup mutation."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import host_observer
flag=Path(sys.argv[1]);mode=sys.argv[2];original=host_observer.snapshot

def snapshot(*args,**kwargs):
    row=original(*args,**kwargs)
    if flag.exists():
        counts=dict(row['oom_events']);counts['oom']+=1;counts['oom_kill']+=1
        row['oom_events']=counts;row['raw']['events']=''.join(f'{key} {value}\n' for key,value in counts.items())
        if mode=='malformed':row['raw']['events']='malformed events\n'
    return row
host_observer.snapshot=snapshot
sys.argv=[str(Path(host_observer.__file__)),*sys.argv[4:]]
raise SystemExit(host_observer.main())
