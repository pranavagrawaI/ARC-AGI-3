"""Unchanged bounded observer/query methods, candidate-only initial boundary label."""
import hashlib,json,os,sys,time
from pathlib import Path
import phase_tools,artifacts,capture,host_observer
from phase_tools import require
from core import shutdown,resources
HERE=Path(__file__).resolve().parent
class Observation:
    def preflight_queries(self,end):
        if self.config['mode']=='CPU_FIXTURE_ONLY':return
        commands={}
        for name,argv in [('gpu',capture.GPU_ARGV),('apps',capture.APPS_ARGV)]:
            result=capture.command(argv,argv,self.output/'query-artifacts',min(end,time.monotonic()+10),'FUTURE_REVIEWED_GPU_QUERY',owner=capture.owner_identity())
            commands[name]=result
            self.emit('query_compatibility',name=name,command=result)
            require(result['ok'] and result['cleanup_verified'],'GPU query/filter compatibility failed; no model preparation')
        (self.output/'query-compatibility.json').write_text(json.dumps(commands,indent=2)+'\n')

    def hardware_and_queries(self):
        end=self.deadline(self.config['hardware']['seconds'])
        self.hardware();self.preflight_queries(end)
        require(time.monotonic()<=end,'hardware/query phase exceeded original30s end')

    def start_observer(self):
        self.observer_spec=host_observer.bind(self.config['run_id'])
        self.observer_spec['settings']=self.config['observer']
        path=self.output/'observer-scope.json';path.write_text(json.dumps(self.observer_spec,indent=2)+'\n')
        self.boundary('before_candidate_preparation')
        spec=dict(argv=[sys.executable,str(HERE/'host_observer.py'),str(path.resolve()),str((self.output/'host-series.jsonl').resolve()),
            str((self.output/'observer.stop').resolve()),str(self.end)],seconds=self.config['total_seconds'])
        self.observer=self.start('observer',spec,self.end)
        self.observer_identity=shutdown.read_process(self.observer.pid)['identity']
        self.emit('observer_owned',identity=self.observer_identity,scope=self.observer_spec)

    def check_observer(self):
        if self.observer is not None:
            require(self.observer.poll() is None,'host observer exited prematurely; evidence invalid')
            value=shutdown.read_process(self.observer.pid)
            require(value is not None and value['identity']==self.observer_identity,'observer ownership identity drift')

    def boundary(self,label):
        self.check_observer()
        started=time.monotonic();row=host_observer.snapshot(self.observer_spec,label)
        invalid=[]
        late=row['end']>min(self.work_end,started+self.config['observer']['boundary_timeout_seconds'])
        if late:invalid.append('host boundary deadline')
        if self.initial_events is None:self.initial_events=row['oom_events']
        oom=not all(row['oom_events'][k]==self.initial_events[k] for k in ('oom','oom_kill'))
        if oom:invalid.append('OOM counter increased; stop load test')
        accounting_error=None
        try:row['original140']=resources.cgroup({k:row['raw'][k].strip() if k in ('maximum','current') else row['raw'][k] for k in ('meminfo','maximum','current','stat')})
        except BaseException as error:
            accounting_error=error;invalid.append('original140 accounting failed');row['accounting_error']=repr(error)
        row.update(valid=not invalid,invalidity=invalid)
        # A returned invalid observation is still evidence. Persistence failures
        # remain fatal; never turn a lost row into a late/OOM-only rejection.
        with (self.output/'host-boundaries.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
        self.boundary_count+=1;self.emit('host_boundary',label=label,start=row['start'],end=row['end'],
            original140=row.get('original140',{}).get('eligible'),valid=row['valid'],invalidity=invalid)
        require(not late,'host boundary deadline')
        require(not oom,'OOM counter increased; stop load test')
        if accounting_error is not None:raise accounting_error

    def stop_observer(self):
        if self.observer is None:return
        self.check_observer();(self.output/'observer.stop').write_text('STOP_AFTER_FINAL_BOUNDARY\n')
        end=min(self.end,time.monotonic()+self.config['observer']['stop_seconds'])
        self.wait('observer',self.observer,end)
        self.cleanup([self.observer.pid])
        report=json.loads((self.output/'host-series.result.json').read_text())
        require(report['pass_'] and report['samples']>0,'host series incomplete')
        import hashlib
        raw=(self.output/'host-series.jsonl').read_bytes()
        require(hashlib.sha256(raw).hexdigest()==report['series_sha256'],'observer series changed')
        rows=[json.loads(line) for line in raw.splitlines()]
        require(all(row['oom_events'][k]==self.initial_events[k] for row in rows for k in ('oom','oom_kill')),'sampled OOM counter increase')
        self.observer=None

    def verify_references(self,value,end):
        if isinstance(value,dict):
            if {'path','sha256','bytes','complete'}<=value.keys():artifacts.verify(self.output/'query-artifacts',value,end)
            else:
                for child in value.values():self.verify_references(child,end)
        elif isinstance(value,list):
            for child in value:self.verify_references(child,end)

