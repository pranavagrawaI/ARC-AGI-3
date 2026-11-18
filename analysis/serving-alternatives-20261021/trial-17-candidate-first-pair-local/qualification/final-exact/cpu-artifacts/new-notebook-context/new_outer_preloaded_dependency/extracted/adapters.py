"""Clocked release adapters. GPU commands are captured, CPU-executor substituted."""
import errno
import json
from pathlib import Path
import shutil
import socket
import time

import artifacts
import capture
import runtime_scope
from core import shutdown,resources


class ControllerQueryFailure(RuntimeError):
    def __init__(self,receipt):
        super().__init__('controller query cleanup unverified; receipt worker not started')
        self.controller_query_receipt=dict(kind='unclean_or_failed_query',**receipt)


class ControllerProbeDeadline(shutdown.DeadlineReached):
    def __init__(self,receipt):
        super().__init__('query preparation phase deadline; no unresolved query')
        self.controller_query_receipt=dict(kind='phase_deadline_clean_queries',**receipt)


def snapshot(path,root,label,end):
    start=time.monotonic();path=Path(path);before=path.stat()
    chunks=[];size=0
    with path.open('rb') as stream:
        while True:
            shutdown.before_deadline(end)
            block=stream.read(65536)
            if not block:break
            size+=len(block)
            if size>artifacts.MAX_RAW_BYTES:raise ValueError('snapshot artifact cap')
            chunks.append(block)
    after=path.stat();raw=b''.join(chunks)
    return raw,artifacts.save(root,label,raw,start),artifacts.version(before)==artifacts.version(after)


class Adapters:
    def __init__(self,root,binding,expected_uuid,commands,paths,port,native_path,native_format,end,execution_policy='LOCAL_CPU_ONLY',scope=None):
        if execution_policy!='LOCAL_CPU_ONLY' and scope is None:
            raise ValueError('runtime cgroup/native fresh-output scope not bound; LOCAL_CPU_ONLY required')
        self.root=Path(root);self.binding=binding;self.uuid=expected_uuid;self.commands=commands
        self.paths=paths;self.port=port;self.native_path=native_path;self.native_format=native_format;self.end=end
        self.sequence=0
        self.execution_policy=execution_policy
        self.owner=capture.owner_identity();self.captured={};self.scope=scope
        if scope is not None:runtime_scope.check(scope,binding,paths,native_path)

    def query(self,name,intended):
        if name not in self.captured:raise ValueError('query not prepared by release controller')
        observed=self.captured[name]
        raw=Path(self.root/observed['artifacts']['stdout']['path']).read_bytes()
        if len(raw)>65536:return dict(eligible=False,error='query output exceeds parser bound',command=observed)
        if not observed['ok']:return dict(eligible=False,error='command incomplete/failed',command=observed)
        try:
            text=raw.decode()
            value=resources.gpu(text,self.uuid) if name=='gpu' else dict(eligible=not text.strip())
            value.pop('raw',None)
            return dict(**value,command=observed)
        except (ValueError,UnicodeError) as error:return dict(eligible=False,error=repr(error),command=observed)

    def check_scope(self):
        return runtime_scope.check(self.scope,self.binding,self.paths,self.native_path) if self.scope is not None else None

    def cgroup(self):
        before=self.check_scope()
        raw={};refs={};stable=True
        for name in ('meminfo','maximum','current','stat'):
            data,refs[name],same=snapshot(self.paths[name],self.root,name,self.end)
            stable=stable and same
            if len(data)>65536:raise ValueError('host parser bound')
            raw[name]=data.decode().strip() if name in ('maximum','current') else data.decode()
        after=self.check_scope()
        value=resources.cgroup(raw);value.pop('raw');eligible=value.pop('eligible')
        return dict(eligible=eligible and stable,**value,artifacts=refs,files_stable_during_read=stable,
                    input_paths={k:str(self.paths[k]) for k in ('meminfo','maximum','current','stat')},
                    observer_cgroup=Path('/proc/self/cgroup').read_text(),
                    registered_frontend_cgroup=self.binding['members'][self.binding['frontend']]['cgroup'],
                    input_scope='LIVE_BOUND_CGROUP_AND_FRESH_STDERR' if self.scope else 'SAVED_CPU_FIXTURES_NOT_RUNTIME_BOUND',
                    scope_before=before,scope_after=after,
                    observation='sequential file reads; not an atomic kernel snapshot')

    def disk(self):
        self.check_scope()
        start=time.monotonic();usage=shutil.disk_usage(self.paths['disk'])
        receipt=dict(path=str(self.paths['disk']),free_bytes=usage.free,start=start,end=time.monotonic())
        ref=artifacts.save(self.root,'disk',json.dumps(receipt).encode(),start)
        return dict(eligible=usage.free>=resources.DISK_MIN_BYTES,free_bytes=usage.free,
                    threshold_bytes=resources.DISK_MIN_BYTES,artifact=ref)

    def port_probe(self):
        start=time.monotonic();error=None
        with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            try:sock.bind(('127.0.0.1',self.port));eligible=True
            except OSError as failure:
                eligible=False;error=dict(errno=failure.errno,message=str(failure))
        raw=dict(address=['127.0.0.1',self.port],reuseaddr=True,eligible=eligible,error=error,start=start,end=time.monotonic())
        return dict(eligible=eligible,artifact=artifacts.save(self.root,'port',json.dumps(raw).encode(),start))

    def native(self):
        before=self.check_scope()
        raw,ref,stable=snapshot(self.native_path,self.root,'native',self.end)
        after=self.check_scope()
        identity=shutdown.read_process(self.binding['frontend']);gone=identity is None
        if identity is not None and identity['identity']!=self.binding['members'][self.binding['frontend']]:
            raise ValueError('native frontend identity drift')
        if self.native_format=='CPU_JSON_EVENTS':
            rows=[json.loads(line) for line in raw.decode().splitlines()]
            events=[r['event'] for r in rows if r['pid']==self.binding['frontend']]
            graceful=('watchdog_stopped' in events and 'scheduler_shutdown_dispatched' in events
                      and events.index('watchdog_stopped')<events.index('scheduler_shutdown_dispatched'))
            crash='crash_path' in events;direct=True
        elif self.native_format=='PINNED_SGLANG_TEXT':
            text=raw.decode();graceful='Gracefully exiting... Remaining number of requests' in text
            crash=any(s in text for s in ('SIGQUIT received.','Triggering SIGQUIT for cleanup...'))
            direct=False
        else:raise ValueError('explicit native evidence format')
        return dict(eligible=stable and gone and graceful and not crash,artifact=ref,
                    frontend_exited=gone,graceful_marker=graceful,crash_marker=crash,
                    direct_watchdog_order_observed=direct,log_stable=stable,
                    input_scope='LIVE_BOUND_CGROUP_AND_FRESH_STDERR' if self.scope else 'LOCAL_FIXTURE_OR_HISTORICAL_LOG_NOT_FRESH_RUNTIME_BOUND',
                    scope_before=before,scope_after=after,
                    limitation='text logs do not directly expose watchdog-stop or GPU context destruction')

    def prepare_controller(self,end):
        capture.check_owner(self.owner);self.check_scope();self.captured={}
        for name,intended in [('gpu',capture.GPU_ARGV),('apps',capture.APPS_ARGV)]:
            try:
                self.captured[name]=capture.command(intended,self.commands[name],self.root,min(self.end,end,time.monotonic()+10),self.execution_policy,owner=self.owner)
            except shutdown.DeadlineReached as failure:
                raise ControllerProbeDeadline(dict(owner_pid=self.owner['pid'],commands=self.captured,dispatch_error=repr(failure))) from failure
            except BaseException as failure:
                raise ControllerQueryFailure(dict(owner_pid=self.owner['pid'],commands=self.captured,dispatch_error=repr(failure))) from failure
            if not self.captured[name]['ok'] or not self.captured[name]['cleanup_verified']:
                raise ControllerQueryFailure(dict(owner_pid=self.owner['pid'],commands=self.captured,dispatch_stopped=True))
        return dict(owner_pid=self.owner['pid'],all_queries_cleaned_before_worker=True,commands=self.captured)

    def probes(self):
        adapter=self
        class ControllerPrepared(dict):
            def prepare_controller(self,end):return adapter.prepare_controller(end)
        return ControllerPrepared(gpu=lambda:self.query('gpu',capture.GPU_ARGV),apps=lambda:self.query('apps',capture.APPS_ARGV),
                    cgroup=self.cgroup,disk=self.disk,port=self.port_probe,native=self.native)
