"""One exact candidate load/window with existing phase and query ownership seams."""
import hashlib,json,os,signal,sys,time
from pathlib import Path
import phase_tools,capture,candidate_scope,candidate_resources,resource_observe
import candidate_evidence as evidence
import outcomes,primary_bundle
from observation import Observation
from core import shutdown
from phase_tools import require
HERE=Path(__file__).resolve().parent

class Runner(Observation,phase_tools.PhaseTools):
    def __init__(self,config,output):
        super().__init__(config,output)
        self.observer=None;self.observer_spec=None;self.observer_identity=None
        self.initial_events=None;self.boundary_count=0
        self.server=None;self.binding=None;self.scope=None;self.spawn_identity=None
        self.candidate_ready=False;self.query_cleanup=[];self.stage='preparation'
        self.root_pidfd=None;self.measurements_finalized=False

    def candidate_gate(self,end=None):
        require(time.monotonic()<min(self.work_end,end or self.work_end),'candidate preparation deadline')
        self.check_observer()
        require(not capture.RETAINED and all(g==self.observer.pid for g in self.groups),
                'unexpected query/phase ownership before only candidate load')

    def bind_server(self):
        require(self.server is not None and self.spawn_identity is not None,'candidate root not retained')
        current=shutdown.read_process(self.server.pid)
        require(current is not None and current['identity']['starttime']==self.spawn_identity['starttime']
            and current['identity']['boot_id']==self.spawn_identity['boot_id'],'candidate root reused/gone')
        launched=json.loads((self.output/'candidate-package/candidate-launch.json').read_text())
        choices=[self.config['candidate']['server']['argv'],launched['argv']]
        matches=[argv for argv in choices if hashlib.sha256(b''.join(os.fsencode(s)+b'\0' for s in argv)).hexdigest()==current['identity']['cmdline_sha256']]
        require(len(matches)==1,'candidate argv not wrapper or declared prepared entry')
        sources={str((HERE/name).resolve()):hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                 for name in ('candidate_prepare.py','candidate_library.py','candidate-plan.json','candidate_runner.py')}
        for name in ('candidate-launch.json','installed-source-identity.json','model-source-identity.json','prepared.json'):
            path=(self.output/'candidate-package'/name).resolve();sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        self.binding,self.scope=candidate_scope.bind(self.server,matches[0],self.config['run_id'],sources,self.observer_spec)
        (self.output/'candidate-scope.json').write_text(json.dumps(self.scope,indent=2)+'\n')
        return current['identity']

    def candidate_phase(self):
        a=self.config['candidate']
        for name in ('prepare','identity'):
            self.stage=name;self.boundary('candidate.'+name+'.before')
            self.command('candidate.'+name,a[name]);self.boundary('candidate.'+name+'.after')
        self.stage='startup';startup_end=self.deadline(a['server']['seconds'])
        self.server=self.start('candidate.server',a['server'],self.work_end)
        self.root_pidfd=os.pidfd_open(self.server.pid)
        self.spawn_identity=shutdown.read_process(self.server.pid)['identity']
        self.emit('candidate_root_retained',identity=self.spawn_identity,startup_deadline=startup_end)
        self.boundary('candidate.server.spawned')
        self.ready('candidate',a,self.server,startup_end)
        self.candidate_ready=True;identity=self.bind_server()
        self.emit('candidate_server_bound',identity=identity,session_id=self.config['run_id'],gpu_uuid=self.gpu_uuid)
        self.boundary('candidate.health200')
        self.stage='replay'
        spec=dict(a['replay']);spec['env']=dict(spec.get('env',{}),SCREEN_DRAIN_SECONDS=str(a['drain_seconds']),
            SCREEN_SESSION_ID=self.config['run_id'],SCREEN_GPU_UUID=self.gpu_uuid,
            SCREEN_SERVER_PGID=str(self.server.pid),SCREEN_SERVER_START=str(identity['starttime']))
        self.boundary('candidate.replay.before');self.command('candidate.replay',spec,self.server);self.boundary('candidate.replay.after')
        self.stage='qualify';self.command('candidate.qualify',a['qualify'],self.server)
        self.boundary('candidate.qualify.after')
        self.emit('arm_returned',arm='candidate')

    def finish_candidate(self):
        if self.server is None:return
        if self.binding is None:self.bind_server()
        # Do not rebind a qualified server: unknown/reused entrants stay fail-closed.
        shutdown.owned(self.binding)
        start=time.monotonic();end=min(self.work_end,start+self.config['final_resource_seconds'])
        self.emit('candidate_cleanup_start',cleanup_start=start,phase_deadline=end,server_pgid=self.server.pid)
        receipt=dict(start=start,absolute_deadline=end,server_pgid=self.server.pid,pass_=False,error=None,samples=[])
        try:
            require(time.monotonic()<end,'final180 starts after work end')
            self.cleanup([self.server.pid],deadline=end)
            require(time.monotonic()<=end and not shutdown.owned(self.binding),'candidate cleanup late/ownership unresolved')
            self.boundary('candidate.cleaned',finalizing=True)
            paths={key:Path(value) for key,value in self.scope['paths'].items()}
            adapter=candidate_resources.Adapters(self.output/'query-artifacts',self.binding,self.gpu_uuid,paths,8001,end,self.scope)
            while True:
                self.check_observer();require(time.monotonic()<end,'final180 resource deadline')
                require(not capture.RETAINED and all(g==self.observer.pid for g in self.groups),'unknown phase/query ownership')
                sample=resource_observe.observe(self.binding,adapter.probes(),end,self.config['resource_probe_seconds'])
                receipt['samples'].append(sample);self.verify_references(sample,end)
                if candidate_resources.accepted(sample,end):break
                time.sleep(min(2,max(0,end-time.monotonic())))
            receipt['end']=time.monotonic();require(receipt['end']<=end,'late resource completion')
            receipt['pass_']=True
            self.emit('candidate_resources_pass',start=start,end=receipt['end'],absolute_deadline=end)
        except BaseException as error:
            receipt['error']=repr(error);raise
        finally:
            receipt.setdefault('end',time.monotonic())
            (self.output/'candidate-release.json').write_text(json.dumps(receipt,indent=2)+'\n')

    def emergency(self):
        errors=[]
        # Controller retains pidfds when query cleanup fails; no numeric-PID fallback.
        for token,record in list(capture.RETAINED.items()):
            if record['owner']!=os.getpid():errors.append('foreign query owner');continue
            try:
                require(time.monotonic()<self.end,'query emergency deadline')
                signal.pidfd_send_signal(record['pidfd'],signal.SIGKILL)
                record['child'].wait(timeout=max(0,self.end-time.monotonic()))
                os.close(record['pidfd']);capture.RETAINED.pop(token)
                self.query_cleanup.append(dict(token=token,exited=True,end=time.monotonic()))
            except BaseException as error:errors.append(repr(error))
        if self.server is not None and self.server.pid in self.groups:
            try:
                if self.binding is None:self.bind_server()
                shutdown.owned(self.binding)
                self.cleanup([self.server.pid],deadline=self.end)
            except BaseException as error:errors.append(repr(error))
        # Never pass a rejected candidate identity to generic group cleanup.
        others=[g for g in self.groups if self.server is None or g!=self.server.pid]
        try:self.cleanup(others,deadline=self.end)
        except BaseException as error:errors.append(repr(error))
        return errors

    def run(self):
        result=dict(run_id=self.config['run_id'],purpose=self.config['purpose'],pass_=False,
                    status='INVALID_INCONCLUSIVE',U_resp=None,adoption=False)
        primary=None;cleanup_errors=[];failed_stage=None
        try:
            self.hardware_and_queries();self.start_observer()
            self.boundary('corpus.prepare.before');self.command('corpus.prepare',self.config['corpus_prepare']);self.boundary('corpus.prepare.after')
            self.candidate_phase()
        except BaseException as error:
            primary=error;failed_stage=self.stage
            result.update(error=repr(error),failed_stage=failed_stage)
        finally:
            try:self.finish_candidate()
            except BaseException as error:cleanup_errors.append(repr(error))
            if self.observer_spec is not None:
                try:self.boundary('final_cleanup',finalizing=True)
                except BaseException as error:cleanup_errors.append(repr(error))
                # Preserve available observer completion even if the final boundary is invalid.
                try:self.stop_observer(finalizing=True);self.measurements_finalized=True
                except BaseException as error:cleanup_errors.append(repr(error))
            cleanup_errors.extend(self.emergency())
        result.update(cleanup_errors=cleanup_errors,phase_cleanup_errors=self.cleanup_failures,
            surviving_phase_groups=list(self.groups),retained_queries=list(capture.RETAINED),query_cleanup=self.query_cleanup,
            candidate_health200=self.candidate_ready,elapsed_seconds=time.monotonic()-self.started,
            total_seconds=self.config['total_seconds'],host_boundaries=self.boundary_count)
        try:
            require(not cleanup_errors and not self.cleanup_failures and not self.groups and not capture.RETAINED,'cleanup incomplete')
            end=self.deadline(self.config['verify_seconds'])
            proof=evidence.check(self.output,self.config,primary_error=repr(primary) if primary else None,failed_stage=failed_stage)
            require(primary is None or proof['status']!='CANDIDATE_ALONE_WORKLOAD_FEASIBLE','primary error cannot be a feasible screen')
            require(time.monotonic()<=end,'final verification deadline')
            result.update(proof);result['pass_']=proof['status']=='CANDIDATE_ALONE_WORKLOAD_FEASIBLE'
        except BaseException as error:result['verification_error']=repr(error)
        if time.monotonic()>self.work_end:result.update(pass_=False,status='INVALID_INCONCLUSIVE',U_resp=None,late_work=True)
        result['elapsed_seconds']=time.monotonic()-self.started
        self.emit('terminal',**result)
        (self.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        self.trace.close()
        if self.root_pidfd is not None:os.close(self.root_pidfd)
        for handle in self.handles:handle.close()
        try:
            publication_end=self.deadline(self.config['primary_seconds'])
            publication=primary_bundle.write(self.output,self.config,publication_end)
            returned_at=time.monotonic()
            require(publication['start']<=publication['end']<=publication['persisted_end']<=returned_at<=publication_end<=self.work_end,'primary publication returned late')
            receipt_bytes=(self.output/'primary-bundle-receipt.json').read_bytes()
            completion=dict(schema='PRIMARY_CALLER_COMPLETION_V2',helper_persisted_end=publication['persisted_end'],returned_at=returned_at,
                deadline=publication_end,receipt_sha256=hashlib.sha256(receipt_bytes).hexdigest())
            (self.output/'primary-publication-completion.json').write_text(json.dumps(completion,indent=2)+'\n')
            require(time.monotonic()<=publication_end,'primary completion persistence returned late')
        except BaseException as error:
            result.update(pass_=False,status='INVALID_INCONCLUSIVE',U_resp=None,primary_bundle_error=repr(error))
            (self.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        return result
