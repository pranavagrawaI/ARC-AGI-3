"""Independent notebook watchdog; scopes termination to the kernel's children."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import time


def process(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(') ',1)[1].split()
        return {'pid':pid,'ppid':int(fields[1]),'start':int(fields[19]),'state':fields[0]}
    except (OSError,ValueError,IndexError): return None


def descendants(parent,exclude):
    entries={int(p.name):process(int(p.name)) for p in Path('/proc').iterdir() if p.name.isdecimal()}
    selected={parent}
    while True:
        added={pid for pid,item in entries.items() if item and item['ppid'] in selected
               and pid not in selected and pid!=exclude}
        if not added: break
        selected.update(added)
    return [entries[pid] for pid in selected if pid!=parent and entries.get(pid)]


def send(item,sig):
    current=process(item['pid'])
    if current and current['start']==item['start']:
        try: os.kill(item['pid'],sig)
        except ProcessLookupError: pass


def run(parent,root,*,started,soft_seconds,hard_seconds,force_margin=10):
    root=Path(root); identity=process(parent)
    if identity is None: return
    receipt={'parent':identity,'soft_seconds':soft_seconds,'hard_seconds':hard_seconds,'events':[]}
    def record(event,**data):
        receipt['events'].append({'event':event,'elapsed_seconds':time.monotonic()-started,**data})
        (root/'watchdog-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    soft_sent=False
    while True:
        current=process(parent)
        if current is None or current['start']!=identity['start'] or current['state']=='Z':
            record('parent_exited'); return
        if (root/'qualification-complete.json').exists():
            record('notebook_completed'); return
        elapsed=time.monotonic()-started
        if elapsed>=soft_seconds and not soft_sent:
            runner=root/'qualification-runner.json'
            if runner.exists():
                item=json.loads(runner.read_text()); send(item,signal.SIGTERM)
            record('soft_stop_requested',runner_present=runner.exists()); soft_sent=True
        if elapsed>=hard_seconds-force_margin:
            owned=descendants(parent,os.getpid())
            record('forced_stop',child_pids=[p['pid'] for p in owned])
            for item in reversed(owned): send(item,signal.SIGKILL)
            send(identity,signal.SIGKILL)
            return
        time.sleep(min(.2,max(.01,hard_seconds-force_margin-elapsed)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=int,required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--started',type=float,required=True)
    parser.add_argument('--soft',type=float,required=True)
    parser.add_argument('--hard',type=float,required=True)
    parser.add_argument('--force-margin',type=float,default=10)
    args=parser.parse_args()
    if not 0<args.soft<args.hard-args.force_margin: raise ValueError('invalid watchdog allocation')
    run(args.parent,args.root,started=args.started,soft_seconds=args.soft,
        hard_seconds=args.hard,force_margin=args.force_margin)


if __name__=='__main__': main()
