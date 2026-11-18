"""Shipping native methods, isolated from candidate process binding."""
import json,os,signal,time
from pathlib import Path
from urllib.parse import urlsplit
import adapters,capture,registration
from phase_tools import require
from core import shutdown
HERE=Path(__file__).resolve().parent
class NativeSupport:
    def native_env(self,end):
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
            self.native_binding=registration.bind(self.native,self.output/'control-package/control-frontend-registration.json',self.expected)
            return self.native_binding

    def make_adapter(self,commands,paths,log,scope,end):
            return adapters.Adapters(self.output/'query-artifacts',self.native_binding,self.gpu_uuid,commands,paths,
                urlsplit(self.config['control']['ready_url']).port,log,'PINNED_SGLANG_TEXT',end,'FUTURE_REVIEWED_GPU_QUERY',scope=scope)

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
                    if self.native_binding is None and self.native.poll() is None:self.bind_native()
                    if self.native_binding is not None:
                        living=shutdown.owned(self.native_binding)
                        if living:shutdown.signal_owned(self.native_binding,list(living),signal.SIGKILL,[],self.end)
                        while shutdown.owned(self.native_binding) and time.monotonic()<self.end:time.sleep(.02)
                        require(not shutdown.owned(self.native_binding),'owned native cleanup incomplete')
                    self.native.wait(timeout=max(0,self.end-time.monotonic()))
                except BaseException as error:failures.append(repr(error))
            return failures
