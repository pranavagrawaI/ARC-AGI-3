"""One control load, health200, frontend-first release. No replay/candidate/SDK."""
import argparse,json,os,signal,socket,subprocess,sys,time
from pathlib import Path
from urllib.parse import urlsplit
import phase_tools
from phase_tools import require,positive,PhaseTools
import adapters,artifacts,capture,registration,runtime_scope
from core import shutdown
HERE=Path(__file__).resolve().parent


def validate(config,check_cwd=True):
    require(config['mode'] in ('GPU_SCREEN','CPU_FIXTURE_ONLY'),'mode')
    require(config['purpose']=='CONTROL_POST_LOAD_RELEASE' and config['arm_order']==['control'],'control-only purpose')
    require(not set(config)&{'candidate','replay','corpus_prepare','paired_reduce'},'extra work forbidden')
    require(Path(config['cwd']).is_absolute() and (not check_cwd or Path(config['cwd']).is_dir()),'absolute existing cwd')
    require(positive(config['total_seconds']) and positive(config['cleanup_seconds']) and config['total_seconds']>config['cleanup_seconds'],'absolute cap')
    a=config['control']
    for name in ('prepare','identity','server'):
        spec=a[name];require(positive(spec['seconds']) and spec['argv'] and all(isinstance(v,str) and v for v in spec['argv']),'phase')
    if config['mode']=='GPU_SCREEN':
        require((config['total_seconds'],config['cleanup_seconds'],config['platform_seconds'])==(3600,300,3900),'caps changed')
        require(config['hardware']['argv']==phase_tools.GPU_ARGV and config['hardware']['seconds']==30 and config['hardware']['minimum_memory_mib']==90000,'hardware guard')
        require((a['prepare']['seconds'],a['identity']['seconds'],a['server']['seconds'],a['ready_seconds'])==(480,60,2400,2400),'phase caps')
        require((a['request_seconds'],a['ready_poll_seconds'],a['ready_retry_http_statuses'])==(25,2,[503]),'health policy')
        require(config['release']==dict(seconds=180,graceful_seconds=120,fallback_reserve=60,probe_seconds=25,poll=2),'release budget')
        require(config['verify_seconds']==60,'verify cap')
    else:require(config['total_seconds']<=600,'CPU invocation cap')
    require(urlsplit(a['ready_url']).hostname=='127.0.0.1' and urlsplit(a['ready_url']).port is not None,'loopback readiness')
    return config


class Runner(PhaseTools):
    def __init__(self,config,output):
        super().__init__(config,output)
        self.native=None;self.binding=None;self.adapter=None;self.lease=None;self.expected=None
        self.native_handles=[];self.query_cleanup=[]

    def env(self,end):
        a=self.config['control']
        return dict(os.environ,**a['server'].get('env',{}),SCREEN_PHASE='control.server',SCREEN_RUN_ID=self.config['run_id'],
            SCREEN_OUTPUT=str((self.output/'control-package').resolve()),SCREEN_DEADLINE_MONOTONIC=str(end),
            SCREEN_OVERALL_DEADLINE_MONOTONIC=str(self.end))

    def runtime_contract(self):
        package=self.output/'control-package';prepared=package/'control-prepared.json'
        source_record=json.loads((package/'control-source-identity.json').read_text())
        plan=json.loads((HERE/'control-plan.json').read_text());data=json.loads(prepared.read_text())
        require(source_record['installed_source_sha256']==plan['installed_source_sha256'],'installed source identity')
        sites=list(Path(data['python']).parent.parent.glob('lib/python*/site-packages'))
        require(len(sites)==1,'one installed native site')
        sources={str(sites[0]/name):digest for name,digest in plan['installed_source_sha256'].items()}
        self.expected=registration.contract(prepared,sources,self.config['run_id'])
        return self.expected

    def bind_native(self):
        require(self.native is not None and self.expected is not None,'native expected identity unavailable')
        self.binding=registration.bind(self.native,self.output/'control-package/control-frontend-registration.json',self.expected)
        return self.binding

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

    def native_phase(self):
        a=self.config['control'];package=self.output/'control-package';package.mkdir(exist_ok=True)
        for phase in ('prepare','identity'):self.command('control.'+phase,a[phase])
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
        self.bind_native()
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
            graceful_seconds=spec['graceful_seconds'],fallback_reserve=spec['fallback_reserve'],probe_seconds=spec['probe_seconds'],poll=spec['poll'])
        (self.output/'release.json').write_text(json.dumps(result,indent=2)+'\n')
        self.emit('release_result',**result)
        require(result['pass'],'release predicates not qualified')
        verify_end=self.deadline(self.config['verify_seconds'])
        self.verify_references(before,verify_end);self.verify_references(result,verify_end)
        require(not capture.RETAINED,'unresolved query ownership')
        require(self.native.poll() is not None,'wrapper not reaped')
        self.emit('verified',qualification='POST_LOAD_RELEASE_ONLY')

    def make_adapter(self,commands,paths,log,scope,end):
        return adapters.Adapters(self.output/'query-artifacts',self.binding,self.gpu_uuid,commands,paths,
            urlsplit(self.config['control']['ready_url']).port,log,'PINNED_SGLANG_TEXT',end,'FUTURE_REVIEWED_GPU_QUERY',scope=scope)

    def verify_references(self,value,end):
        if isinstance(value,dict):
            if {'path','sha256','bytes','complete'}<=value.keys():artifacts.verify(self.output/'query-artifacts',value,end)
            else:
                for child in value.values():self.verify_references(child,end)
        elif isinstance(value,list):
            for child in value:self.verify_references(child,end)

    def cleanup_native(self):
        failures=[]
        for token,held in list(capture.RETAINED.items()):
            try:
                require(held['owner']==os.getpid() and held['pidfd'] is not None,'unknown retained query ownership')
                shutdown.before_deadline(self.end)
                if held['child'].poll() is None:signal.pidfd_send_signal(held['pidfd'],signal.SIGKILL)
                held['child'].wait(timeout=max(0,self.end-time.monotonic()));os.close(held['pidfd']);del capture.RETAINED[token]
                self.query_cleanup.append(dict(token=token,original_deadline=held['absolute_deadline'],cleanup_end=time.monotonic(),late_cleanup_not_qualification=True))
            except BaseException as error:failures.append(repr(error))
        if self.native is not None:
            try:
                if self.binding is None and self.native.poll() is None:self.bind_native()
                if self.binding is not None:
                    living=shutdown.owned(self.binding)
                    if living:shutdown.signal_owned(self.binding,list(living),signal.SIGKILL,[],self.end)
                    while shutdown.owned(self.binding) and time.monotonic()<self.end:time.sleep(.02)
                    require(not shutdown.owned(self.binding),'owned native cleanup incomplete')
                self.native.wait(timeout=max(0,self.end-time.monotonic()))
            except BaseException as error:failures.append(repr(error))
        return failures

    def run(self):
        result=dict(run_id=self.config['run_id'],purpose=self.config['purpose'],pass_=False,qualification='NOT_QUALIFIED')
        try:
            self.hardware_and_queries();self.native_phase()
            result.update(pass_=True,qualification='POST_LOAD_RELEASE_ONLY')
        except BaseException as error:result['error']=repr(error)
        finally:
            errors=self.cleanup_native()
            try:self.cleanup()
            except BaseException as error:errors.append(repr(error))
            if errors or self.cleanup_failures or capture.RETAINED:
                result.update(pass_=False,qualification='NOT_QUALIFIED',cleanup_errors=errors,phase_cleanup_errors=self.cleanup_failures)
            result.update(elapsed_seconds=time.monotonic()-self.started,cooperative_cap_seconds=self.config['total_seconds'],
                surviving_phase_groups=list(self.groups),native_pgid=self.native.pid if self.native else None,
                surviving_native_group=phase_tools.group_members(self.native.pid) if self.native else [],
                retained_queries=list(capture.RETAINED),query_cleanup=self.query_cleanup)
            if result['surviving_native_group'] or time.monotonic()>self.end:result.update(pass_=False,qualification='NOT_QUALIFIED')
            self.emit('terminal',**result);(self.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
            for handle in self.handles+self.native_handles:handle.close()
            self.trace.close()
        return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('config');parser.add_argument('output');parser.add_argument('--assignment',required=True);args=parser.parse_args()
    config=validate(json.loads(Path(args.config).read_text()));assignment=json.loads(Path(args.assignment).read_text())
    require(assignment['launch_authorized'] is True and assignment['local_only'] is False and assignment['serialized_assignment'],'real serialized authority required')
    import hashlib
    require(assignment['config_sha256']==hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest() and assignment['total_seconds']==3600 and assignment['platform_hard_timeout_s']==3900,'authority config/caps')
    def stop(signum,frame):raise RuntimeError('signal '+str(signum))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    value=Runner(config,args.output).run();print(json.dumps(value),flush=True);return 0 if value['pass_'] else 1

if __name__=='__main__':raise SystemExit(main())
