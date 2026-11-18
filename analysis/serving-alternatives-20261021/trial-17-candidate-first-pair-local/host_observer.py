"""Read-only, bounded cgroup/process observer. No GPU/native imports or commands.

No peak reset. memory.peak is cumulative, samples miss spikes, RSS shares pages.
This observer is a controller-owned phase child; its failure invalidates evidence.
"""
import argparse,hashlib,json,os,select,time
from pathlib import Path
import runtime_scope
from core import shutdown

MAX_INPUT=65536
MAX_SERIES=256*1024**2

def small(path):
    with Path(path).open('rb') as stream:raw=stream.read(MAX_INPUT+1)
    if len(raw)>MAX_INPUT:raise ValueError('observer input cap')
    return raw.decode()

def bind(run_id):
    observed=runtime_scope.observer();mount=small('/proc/self/mountinfo')
    paths,line=runtime_scope.derive(observed['cgroup'],mount)
    directory=paths['current'].parent
    paths.update(events=directory/'memory.events',events_local=directory/'memory.events.local',peak=directory/'memory.peak',procs=directory/'cgroup.procs')
    required=('meminfo','maximum','current','stat','events','procs')
    keys={name:runtime_scope.key(path) for name,path in paths.items() if name in required}
    return dict(run_id=run_id,controller=shutdown.read_process(os.getpid())['identity'],observer_scope=observed,mount_record=line,
        paths={name:str(path) for name,path in paths.items()},file_keys=keys,required=list(required),bound_at=time.monotonic())

def check(spec):
    controller=shutdown.read_process(spec['controller']['pid'])
    if controller is None or controller['identity']!=spec['controller']:raise ValueError('observer controller identity drift/gone')
    if os.getpid()!=controller['pid'] and os.getppid()!=controller['pid']:raise ValueError('observer not controller/direct child')
    seen=runtime_scope.observer()
    if seen!=spec['observer_scope']:raise ValueError('observer cgroup/namespace drift')
    paths,line=runtime_scope.derive(seen['cgroup'],small('/proc/self/mountinfo'))
    if line!=spec['mount_record'] or any(str(paths[name])!=spec['paths'][name] for name in paths):raise ValueError('observer cgroup mount drift')
    if any(runtime_scope.key(spec['paths'][name])!=key for name,key in spec['file_keys'].items()):raise ValueError('observer input identity drift')

def snapshot(spec,label,include_processes=True,include_io=False):
    started=time.monotonic();check(spec);raw={};clocks={};optional={}
    for name in spec['required']+['peak','events_local']:
        start=time.monotonic()
        try:raw[name]=small(spec['paths'][name])
        except FileNotFoundError:
            if name in spec['required']:raise
            optional[name]='unavailable'
        clocks[name]=dict(start=start,end=time.monotonic())
    if not raw['current'].strip().isdigit() or (raw['maximum'].strip()!='max' and not raw['maximum'].strip().isdigit()):raise ValueError('invalid original cgroup capacity/current')
    events={k:int(v) for k,v in (line.split() for line in raw['events'].splitlines())}
    if not {'oom','oom_kill'}<=events.keys():raise ValueError('complete OOM counters required')
    pids=[int(s) for s in raw['procs'].split()]
    if len(pids)>512:raise ValueError('process observation bound exceeded')
    processes=[]
    for pid in pids if include_processes else []:
        begin=time.monotonic()
        try:
            value=shutdown.read_process(pid)
            if value is None:processes.append(dict(pid=pid,gone_during_read=True,start=begin,end=time.monotonic()));continue
            status=small(Path('/proc')/str(pid)/'status')
            after=shutdown.read_process(pid)
            if after is not None and after['identity']!=value['identity']:raise shutdown.UnsafeIdentity('process changed during diagnostic observation')
            processes.append(dict(**value,status=status,gone_after_read=after is None,start=begin,end=time.monotonic()))
        except (FileNotFoundError,ProcessLookupError,shutdown.UnsafeIdentity) as failure:
            # Diagnostic RSS rows do not grant signal/admission authority. Retain
            # incomplete identities; do not invent RSS/peak/ownership for them.
            processes.append(dict(pid=pid,diagnostic_unavailable=repr(failure),start=begin,end=time.monotonic()))
    io={}
    if include_io:
        for name,path in [('host_io_pressure',Path('/proc/pressure/io')),('cgroup_io',Path(spec['paths']['current']).parent/'io.stat')]:
            begin=time.monotonic()
            try:io[name]=dict(raw=small(path),start=begin,end=time.monotonic())
            except OSError as error:io[name]=dict(unavailable=repr(error),start=begin,end=time.monotonic())
    check(spec)
    return dict(io_diagnostics=io,label=label,start=started,end=time.monotonic(),run_id=spec['run_id'],observer_pid=os.getpid(),scope=spec['observer_scope'],
        paths=spec['paths'],raw=raw,optional=optional,file_clocks=clocks,oom_events=events,processes=processes,process_diagnostics_sampled=include_processes,
        limitations=['sequential non-atomic reads','samples miss peaks','memory.peak cumulative and never reset; may include earlier control',
            'RSS/HWM includes shared pages and is not incremental cgroup peak; unavailable process rows preclude complete process-derived accounting','visible cgroup.procs limited by observer namespace','active_file not reclaimability proof'])

def main():
    p=argparse.ArgumentParser();p.add_argument('spec');p.add_argument('output');p.add_argument('stop');p.add_argument('deadline',type=float);a=p.parse_args()
    spec=json.loads(Path(a.spec).read_text());out=Path(a.output);settings=spec['settings'];started=time.monotonic();count=0;size=0
    ownerfd=os.pidfd_open(spec['controller']['pid']);error=None;clean=False
    try:
        with out.open('x') as stream:
            while time.monotonic()<a.deadline:
                if select.select([ownerfd],[],[],0)[0]:raise ValueError('observer owner exited')
                row=snapshot(spec,'sample',include_processes=count%settings['process_every_samples']==0,include_io=count%20==0);data=json.dumps(row)+'\n';size+=len(data.encode())
                if size>MAX_SERIES or count>=settings['max_samples']:raise ValueError('observer series bound exceeded')
                stream.write(data);stream.flush();count+=1
                if Path(a.stop).exists():clean=True;break
                time.sleep(min(settings['interval_seconds'],max(0,a.deadline-time.monotonic())))
            if not clean:raise TimeoutError('observer absolute deadline reached')
    except BaseException as failure:error=repr(failure)
    finally:
        os.close(ownerfd)
        receipt=dict(start=started,end=time.monotonic(),pass_=clean and error is None,samples=count,error=error,
            series_sha256=hashlib.sha256(out.read_bytes()).hexdigest() if out.exists() else None,bytes=out.stat().st_size if out.exists() else 0)
        with out.with_suffix('.result.json').open('x') as stream:json.dump(receipt,stream,indent=2)
    return 0 if receipt['pass_'] else 1
if __name__=='__main__':raise SystemExit(main())
