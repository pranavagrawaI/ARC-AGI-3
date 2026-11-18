"""Clocked release adapters. GPU commands are captured, CPU-executor substituted."""
import errno
import json
from pathlib import Path
import shutil
import socket
import time

import artifacts
import capture
from core import shutdown,resources


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
    def __init__(self,root,binding,expected_uuid,commands,paths,port,native_path,native_format,end,execution_policy='LOCAL_CPU_ONLY'):
        self.root=Path(root);self.binding=binding;self.uuid=expected_uuid;self.commands=commands
        self.paths=paths;self.port=port;self.native_path=native_path;self.native_format=native_format;self.end=end
        self.sequence=0
        self.execution_policy=execution_policy

    def query(self,name,intended):
        observed=capture.command(intended,self.commands[name],self.root,min(self.end,time.monotonic()+10),self.execution_policy)
        raw=Path(self.root/observed['artifacts']['stdout']['path']).read_bytes()
        if len(raw)>65536:return dict(eligible=False,error='query output exceeds parser bound',command=observed)
        if not observed['ok']:return dict(eligible=False,error='command incomplete/failed',command=observed)
        try:
            text=raw.decode()
            value=resources.gpu(text,self.uuid) if name=='gpu' else dict(eligible=not text.strip())
            value.pop('raw',None)
            return dict(**value,command=observed)
        except (ValueError,UnicodeError) as error:return dict(eligible=False,error=repr(error),command=observed)

    def cgroup(self):
        raw={};refs={};stable=True
        for name in ('meminfo','maximum','current','stat'):
            data,refs[name],same=snapshot(self.paths[name],self.root,name,self.end)
            stable=stable and same
            if len(data)>65536:raise ValueError('host parser bound')
            raw[name]=data.decode().strip() if name in ('maximum','current') else data.decode()
        value=resources.cgroup(raw);value.pop('raw');eligible=value.pop('eligible')
        return dict(eligible=eligible and stable,**value,artifacts=refs,files_stable_during_read=stable,
                    input_paths={k:str(self.paths[k]) for k in ('meminfo','maximum','current','stat')},
                    observer_cgroup=Path('/proc/self/cgroup').read_text(),
                    registered_frontend_cgroup=self.binding['members'][self.binding['frontend']]['cgroup'],
                    observation='sequential file reads; not an atomic kernel snapshot')

    def disk(self):
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
        raw,ref,stable=snapshot(self.native_path,self.root,'native',self.end)
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
                    limitation='text logs do not directly expose watchdog-stop or GPU context destruction')

    def probes(self):
        return dict(gpu=lambda:self.query('gpu',capture.GPU_ARGV),apps=lambda:self.query('apps',capture.APPS_ARGV),
                    cgroup=self.cgroup,disk=self.disk,port=self.port_probe,native=self.native)
