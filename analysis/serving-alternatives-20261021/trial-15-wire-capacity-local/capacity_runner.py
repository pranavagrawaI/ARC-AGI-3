"""Capacity orchestration using unchanged native/query ownership; no SDK."""
import json,os,socket,subprocess,sys,time
from pathlib import Path
from urllib.parse import urlsplit
import phase_tools,registration,runtime_scope,admission,host_observer,artifacts,capture
from phase_tools import require
from core import shutdown
import release_core
HERE=Path(__file__).resolve().parent
class Runner(release_core.Runner):
    def native_phase(self):
        a=self.config['control'];package=self.output/'control-package';package.mkdir(exist_ok=True)
        for phase in ('prepare','identity'):
            self.boundary('control.'+phase+'.before');self.command('control.'+phase,a[phase]);self.boundary('control.'+phase+'.after')
        self.runtime_contract()
        with socket.socket() as check:
            check.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);check.bind(('127.0.0.1',urlsplit(a['ready_url']).port))
        startup_end=self.deadline(a['server']['seconds'])
        parent=self.output/'native';parent.mkdir(exist_ok=True)
        self.lease=runtime_scope.create_native_output(parent.resolve(),self.config['run_id'])
        stdout=(self.output/'control.server.stdout').open('xb');self.native_handles.append(stdout)
        with runtime_scope.open_native_stderr(self.lease) as stderr:
            launched=time.monotonic()
            self.native=subprocess.Popen(a['server']['argv'],cwd=self.config['cwd'],env=self.env(startup_end),
                stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True)
        # Native server is intentionally NOT in PhaseTools.groups: no group TERM.
        self.emit('native_spawn',pid=self.native.pid,argv=a['server']['argv'],startup_deadline=startup_end)
        registration_end=min(startup_end,launched+10)
        receipt=self.output/'control-package/control-frontend-registration.json'
        while not receipt.exists():
            require(self.native.poll() is None,'wrapper died before registration')
            require(time.monotonic()<registration_end,'native registration deadline')
            time.sleep(.01)
        runtime_scope.record_wrapper(self.native,self.lease,launched)
        (self.output/'native-lease.json').write_text(json.dumps(self.lease,indent=2)+'\n')
        self.ready('control',a,self.native,startup_end)
        self.boundary('control.health200')
        self.bind_native()
        self.replay_and_qualify('control',self.native)
        paths,_=runtime_scope.derive(Path('/proc/self/cgroup').read_text(),Path('/proc/self/mountinfo').read_text())
        log=Path(self.lease['created']['log_path']);scope=runtime_scope.bind(self.binding,paths,self.lease,self.config['run_id'])
        (self.output/'release-scope.json').write_text(json.dumps(scope,indent=2)+'\n')
        commands={'gpu':capture.GPU_ARGV,'apps':capture.APPS_ARGV}
        spec=self.config['release'];release_end=self.deadline(spec['seconds'])
        self.adapter=self.make_adapter(commands,paths,log,scope,release_end)
        before=shutdown.observe(self.binding,self.adapter.probes(),release_end,spec['probe_seconds'])
        (self.output/'pre-release.json').write_text(json.dumps(before,indent=2)+'\n')
        self.emit('release_start',deadline=release_end)
        result=shutdown.release(self.binding,self.adapter.probes(),end=release_end,
            graceful_seconds=spec['graceful_seconds'],fallback_reserve=spec['fallback_reserve'],probe_seconds=spec['probe_seconds'],poll=spec['poll'],admission_policy='FIRST_FRESH_NONHOST')
        (self.output/'release.json').write_text(json.dumps(result,indent=2)+'\n')
        self.emit('release_result',**result)
        require(result['pass'],'nonhost release not qualified; no candidate')
        verify_end=self.deadline(self.config['verify_seconds'])
        self.verify_references(before,verify_end);self.verify_references(result,verify_end)
        require(not capture.RETAINED,'unresolved query ownership')
        require(self.native.poll() is not None,'wrapper not reaped')
        require(not shutdown.owned(self.binding),'native ownership unresolved')
        self.boundary('control.released.before_admission')
        fresh_end=min(release_end,self.deadline(self.config['admission']['fresh_probe_seconds']))
        fresh=shutdown.observe(self.binding,self.adapter.probes(),fresh_end,self.config['admission']['fresh_probe_seconds'])
        self.verify_references(fresh,fresh_end)
        self.candidate_gate(fresh=fresh)
        receipt=dict(sample=fresh,original140_result=fresh['probes']['cgroup']['eligible'],original_all_predicates=fresh['eligible'],
            policy='FIRST_FRESH_NONHOST',no_numeric_host_admission=True,admitted_at=time.monotonic())
        (self.output/'candidate-admission.json').write_text(json.dumps(receipt,indent=2)+'\n')
        self.candidate_gate(fresh=fresh)
        self.check_observer()
        self.emit('candidate_admitted',**receipt)
        self.candidate_admission=fresh
        self.candidate_phase()

    def run(self):
        result=dict(run_id=self.config['run_id'],purpose=self.config['purpose'],pass_=False,qualification='FEASIBILITY_ONLY')
        try:
            self.hardware_and_queries();self.start_observer()
            self.boundary('corpus.prepare.before');self.command('corpus.prepare',self.config['corpus_prepare']);self.boundary('corpus.prepare.after')
            self.native_phase()
            require(time.monotonic()<=self.work_end,'late candidate cleanup/resource phase')
            self.boundary('final_cleanup');self.stop_observer();self.measurements_finalized=True
            self.command('pair.reduce',self.config['paired_reduce'])
            pair=json.loads((self.output/'paired-result.json').read_text())
            result.update(pass_=pair['verdict']!='FEASIBILITY_ONLY',qualification=pair['verdict'],ratio=pair['ratio'],runtime_completed=True)
        except BaseException as error:result['error']=repr(error)
        finally:
            errors=self.cleanup_native()
            if self.observer_spec is not None and not getattr(self,'measurements_finalized',False):
                try:self.boundary('final_cleanup');self.stop_observer()
                except BaseException as error:errors.append(repr(error))
            try:self.cleanup()
            except BaseException as error:errors.append(repr(error))
            if errors or self.cleanup_failures or capture.RETAINED:
                result.update(pass_=False,qualification='FEASIBILITY_ONLY',cleanup_errors=errors,phase_cleanup_errors=self.cleanup_failures)
            result.update(elapsed_seconds=time.monotonic()-self.started,cooperative_cap_seconds=self.config['total_seconds'],
                surviving_phase_groups=list(self.groups),native_pgid=self.native.pid if self.native else None,
                surviving_native_group=phase_tools.group_members(self.native.pid) if self.native else [],
                retained_queries=list(capture.RETAINED),query_cleanup=self.query_cleanup,candidate_health200=self.candidate_ready,host_boundaries=self.boundary_count)
            if result['surviving_native_group'] or time.monotonic()>self.end:result.update(pass_=False,qualification='FEASIBILITY_ONLY')
            self.emit('terminal',**result);(self.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
            for handle in self.handles+self.native_handles:handle.close()
            self.trace.close()
        return result

    def replay_and_qualify(self,arm,server):
        identity=shutdown.read_process(server.pid)
        require(identity is not None and identity['identity']['pgid']==server.pid,'actual replay server identity')
        bound=dict(arm=arm,identity=identity['identity'],session_id=self.config['run_id'],gpu_uuid=self.gpu_uuid,
            registered_native_binding=self.binding if arm=='control' else None)
        if arm=='control':require(identity['identity']==self.binding['members'][server.pid],'native wrapper drift')
        (self.output/(arm+'-server-binding.json')).write_text(json.dumps(bound,indent=2)+'\n')
        self.emit('capacity_server_bound',**bound)
        a=self.config[arm]
        for phase in ('replay','qualify'):
            self.boundary(arm+'.'+phase+'.before');spec=dict(a[phase])
            if phase=='replay':spec['env']=dict(spec.get('env',{}),SCREEN_DRAIN_SECONDS=str(a['drain_seconds']),SCREEN_SESSION_ID=self.config['run_id'],SCREEN_GPU_UUID=self.gpu_uuid,SCREEN_SERVER_PGID=str(server.pid),SCREEN_SERVER_START=str(identity['identity']['starttime']))
            self.command(arm+'.'+phase,spec,server);self.boundary(arm+'.'+phase+'.after')
        self.emit('arm_pass',arm=arm)

    def candidate_phase(self):
        a=self.config['candidate'];self.candidate_gate();self.command('candidate.prepare',a['prepare']);self.boundary('candidate.prepare.after')
        self.command('candidate.identity',a['identity']);self.boundary('candidate.identity.after')
        startup_end=self.deadline(a['server']['seconds']);server=self.start('candidate.server',a['server'],self.work_end)
        primary=None;cleanup_start=None;phase_end=None
        try:
            self.boundary('candidate.server.spawned');self.ready('candidate',a,server,startup_end)
            self.boundary('candidate.health200');self.candidate_ready=True
            self.replay_and_qualify('candidate',server)
            cleanup_start=time.monotonic();phase_end=min(self.work_end,cleanup_start+self.config['final_resource_seconds'])
            self.emit('candidate_cleanup_start',phase_deadline=phase_end,server_pgid=server.pid)
        except BaseException as error:
            primary=error;raise
        finally:
            try:self.cleanup([server.pid],deadline=phase_end)
            except BaseException as error:
                self.cleanup_failures.append(dict(context='candidate.capacity',error=repr(error),primary_error=repr(primary) if primary else None))
                self.emit('cleanup_failure',**self.cleanup_failures[-1])
                if primary is None:raise
        require(phase_end is not None and time.monotonic()<=phase_end,'late candidate cleanup; no pair qualification')
        self.boundary('candidate.cleaned')
        self.final_resources(cleanup_start,phase_end,server.pid)

    def final_resources(self,start,end,server_pgid):
        paths,_=runtime_scope.derive(Path('/proc/self/cgroup').read_text(),Path('/proc/self/mountinfo').read_text())
        scope=json.loads((self.output/'release-scope.json').read_text());log=Path(self.lease['created']['log_path'])
        adapter=self.make_adapter({'gpu':capture.GPU_ARGV,'apps':capture.APPS_ARGV},paths,log,scope,end)
        receipt=dict(start=start,absolute_deadline=end,pass_=False,samples=[],error=None,server_pgid=server_pgid,native_marker_scope='PRIOR_SHIPPING_CONTROL_ONLY')
        try:
            while True:
                self.check_observer();require(time.monotonic()<end,'candidate cleanup/resource deadline')
                require(not phase_tools.group_members(server_pgid) and not capture.RETAINED,'candidate/query ownership unresolved')
                require(not [g for g in self.groups if self.observer is None or g!=self.observer.pid],'phase ownership unresolved')
                sample=shutdown.observe(self.binding,adapter.probes(),end,self.config['release']['probe_seconds']);receipt['samples'].append(sample)
                self.verify_references(sample,end)
                if admission.nonhost(sample,end):break
                time.sleep(min(self.config['release']['poll'],max(0,end-time.monotonic())))
            receipt['end']=time.monotonic();require(receipt['end']<=end,'late final resource observation')
            receipt['pass_']=True;self.emit('candidate_resources_pass',start=start,end=receipt['end'],absolute_deadline=end)
        except BaseException as error:receipt['error']=repr(error);raise
        finally:
            receipt.setdefault('end',time.monotonic());(self.output/'candidate-release.json').write_text(json.dumps(receipt,indent=2)+'\n')
