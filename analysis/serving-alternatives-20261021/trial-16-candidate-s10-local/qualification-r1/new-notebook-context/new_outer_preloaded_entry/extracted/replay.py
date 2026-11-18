"""Ten owned HTTP workers, immutable FIFO streams, one fixed window, raw transport.

No native occupancy or processor inspection. No retry, sleep between jobs, or
post-cutoff admission. Connection/socket shutdown bounds even trickling bodies.
"""
import argparse
import http.client
import json
import os
from pathlib import Path
import socket
import threading
import time
from urllib.parse import urlsplit
import catalog

MAX_RESPONSE=16*1024**2
READ_INACTIVITY_SECONDS=900.

class CutoffCancelled(Exception):
    pass

class PersistenceFailure(OSError):
    pass

def persist(operation,*args):
    """Disk failure stays fatal even while transport is being cancelled."""
    try:return operation(*args)
    except Exception as exc:raise PersistenceFailure(repr(exc)) from exc

class CaptureConnection(http.client.HTTPConnection):
    def __init__(self,*args,trace,deadline,**kwargs):
        super().__init__(*args,**kwargs);self.trace=trace;self.deadline=deadline;self.writes=[]
        self.raw_socket=None;self.cutoff_shutdown_at=None
    def connect(self):
        super().connect();self.raw_socket=self.sock
    def send(self,data):
        if not isinstance(data,bytes):raise ValueError('wire must be exact bytes')
        if self.sock is None:self.connect()
        self.trace.write(data);self.trace.flush()
        now=time.monotonic()
        if now>=self.deadline:raise TimeoutError('no transport write after cutoff')
        self.sock.sendall(data)
        ended=time.monotonic();self.writes.append({'t':now,'ended':ended,'bytes':len(data)})
        if ended>=self.deadline:raise TimeoutError('transport write crossed cutoff; comparison invalid')

def prepare(corpus,destination):
    """Outside replay clock. Whole immutable corpus verified, each payload bound."""
    directory=Path(corpus);destination=Path(destination);destination.mkdir(exist_ok=False)
    catalog.verify_corpus(directory);cat=catalog.load()
    index={}
    for key,record in cat['records'].items():
        body=catalog.payload_for(record,directory)
        if catalog.wire_hash(body)!=record['wire_sha'] or catalog.image_observations(body)!=record['images']:
            raise ValueError('prepared wire/source mismatch '+key)
        raw=catalog.canonical(body);catalog.checked_body(raw,record);name=key.replace(':','-')+'.json'
        (destination/name).write_bytes(raw);index[key]={'file':name,'sha256':catalog.digest(raw)}
    (destination/'index.json').write_bytes(catalog.canonical(index))
    return index

class Replay:
    def __init__(self,cat,bodies,out,url,arm,seconds=(600.,900.),identity=None,slots=10):
        self.cat=cat;self.bodies=Path(bodies);self.out=Path(out);self.out.mkdir(parents=True,exist_ok=False)
        self.url=urlsplit(url);self.arm=arm;self.warmup,self.measure=seconds
        if self.url.scheme!='http' or self.url.hostname not in ('127.0.0.1','localhost'):raise ValueError('local HTTP only')
        self.index=json.loads((self.bodies/'index.json').read_text())
        self.slots=slots
        self.lock=threading.RLock();self.stop=threading.Event();self.ready=threading.Barrier(slots+1)
        self.queue=list(cat['initial_holders'])+list(cat['initial_queue']);self.cursor={g:0 for g in cat['streams']}
        self.holders=set();self.pending={};self.connections={};self.threads=[];self.seq=0;self.serial=0
        self.boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip();self.identity=identity or {}
        self.log=(self.out/'events.jsonl').open('x');self.origin=None;self.failed=False

    def emit(self,kind,t=None,**fields):
        row=dict(seq=self.seq,t=time.monotonic() if t is None else t,kind=kind,pid=os.getpid(),boot_id=self.boot,**fields)
        self.seq+=1;self.log.write(json.dumps(row)+'\n');self.log.flush()

    def job(self,previous=None):
        with self.lock:
            if self.stop.is_set() or time.monotonic()>=self.end:return None
            if previous is not None:
                self.cursor[previous]+=1;i=self.cursor[previous];stream=self.cat['streams'][previous]
                finished=i==len(stream)
                if finished or not self.cat['records'][stream[i]]['extends_previous']:
                    self.holders.remove(previous);self.emit('release',game=previous,reason='finished' if finished else 'handover')
                    if not finished:self.queue.append(previous)
                    previous=None
            if previous is None:
                if not self.queue:return None
                previous=self.queue.pop(0);self.holders.add(previous);self.emit('admit',game=previous)
            key=self.cat['streams'][previous][self.cursor[previous]]
            rid=f'{self.identity.get("session_id","local")}-{self.arm}-{self.serial}'
            self.serial+=1
            return previous,key,rid

    def fail(self,rid,message):
        with self.lock:
            self.failed=True;self.stop.set();self.emit('error',rid=rid,error=message)

    def once(self,key,rid,worker):
        row=self.index[key];body=(self.bodies/row['file']).read_bytes()
        record=self.cat['records'][key]
        catalog.checked_body(body,record)
        if row['sha256']!=record['prepared_body_sha256']:raise ValueError('prepared index/body binding changed')
        d=self.out/rid;d.mkdir()
        with (d/'wire.http').open('xb') as trace:
            conn=CaptureConnection(self.url.hostname,self.url.port,timeout=2,trace=trace,deadline=self.end)
            with self.lock:
                if self.stop.is_set() or time.monotonic()>=self.end:return False
                self.connections[rid]=conn
            complete=False;status=None;response_headers=[];received=0;error=None;body_completed=None;persisted=0
            termination=None;error_kind=None;cause=None;cause_type=None;error_observed_at=None;stage='CONNECT'
            read_timeout=None;phase_at_error=None;outbound_body_sha256=None
            connection_started=time.monotonic();started=None;success=False
            try:
                conn.connect()
                with self.lock:
                    if time.monotonic()>=self.end:raise CutoffCancelled('fixed cutoff before dispatch')
                    if self.stop.is_set():raise RuntimeError('admission stopped by failure')
                    self.pending[rid]=key
                    self.emit('send',rid=rid,key=key,worker=worker,wire_ref=f'{rid}/wire.json',transport_ref=f'{rid}/wire.http')
                started=time.monotonic()
                stage='SEND'
                conn.request('POST','/v1/chat/completions',body=body,
                             headers={'Content-Type':'application/json','X-Request-Id':rid,'Connection':'close'})
                actual=(d/'wire.http').read_bytes().split(b'\r\n\r\n',1)[1]
                catalog.checked_body(actual,record);outbound_body_sha256=catalog.digest(actual)
                # Socket inactivity bound does not replace the absolute clock;
                # parent shutdown still ends slow/trickling headers and bodies.
                # Keep one owner for the absolute cutoff. A second timer armed
                # from end-now can fire before this socket's parent shutdown.
                # This remains a per-blocking-read inactivity bound, NOT an
                # absolute expiry derived from armed_at + actual_seconds.
                arm_started=time.monotonic()
                conn.sock.settimeout(READ_INACTIVITY_SECONDS)
                read_timeout=dict(mode='SOCKET_INACTIVITY',actual_seconds=conn.sock.gettimeout(),
                                  started_at=arm_started,armed_at=time.monotonic(),cutoff=self.end)
                stage='READ_HEADERS'
                response=conn.getresponse();status=response.status;response_headers=response.getheaders()
                with self.lock:
                    if status!=200:self.fail(rid,'HTTP '+str(status))
                stage='PROTOCOL'
                lengths=response.headers.get_all('Content-Length',[])
                if len(lengths)!=1 or not lengths[0].isdigit():raise ValueError('one explicit Content-Length required')
                expected=int(lengths[0])
                if expected>MAX_RESPONSE:raise ValueError('response byte limit')
                output=persist((d/'body.bin').open,'xb')
                try:
                    while received<expected:
                        stage='READ_BODY'
                        chunk=response.read1(min(65536,expected-received))
                        if not chunk:
                            if conn.cutoff_shutdown_at is not None:raise CutoffCancelled('fixed cutoff during body read')
                            raise ValueError('incomplete response body')
                        received+=len(chunk)
                        if received==expected:body_completed=time.monotonic()
                        written=persist(output.write,chunk)
                        if written!=len(chunk):raise PersistenceFailure('short body persistence write')
                        persisted+=written
                finally:persist(output.close)
                complete=True;termination='COMPLETE'
            except Exception as exc:
                error_observed_at=time.monotonic();phase_at_error=stage
                cancelled=isinstance(exc,CutoffCancelled) or (
                    stage in ('CONNECT','SEND','READ_HEADERS','READ_BODY')
                    and conn.cutoff_shutdown_at is not None and body_completed is None
                    and isinstance(exc,(ConnectionResetError,ConnectionAbortedError,BrokenPipeError)))
                if cancelled:
                    termination='INTENTIONAL_CUTOFF';error_kind='INTENTIONAL_CUTOFF'
                    cause=repr(exc);cause_type=type(exc).__name__
                    error=repr(CutoffCancelled('owned socket cancelled at fixed cutoff'))
                else:
                    termination='ERROR';error_kind='PERSISTENCE' if isinstance(exc,PersistenceFailure) else 'TRANSPORT_OR_PROTOCOL'
                    error=repr(exc);self.fail(rid,error)
            finally:
                ended=body_completed if body_completed is not None else time.monotonic();conn.close()
                http=dict(rid=rid,key=key,worker=worker,status=status,headers=response_headers,
                          complete=complete,received_bytes=received,persisted_bytes=persisted,
                          body_completed_at=body_completed,termination=termination,error_kind=error_kind,error=error,cause=cause,cause_type=cause_type,
                          error_observed_at=error_observed_at,cutoff_shutdown_at=conn.cutoff_shutdown_at,
                          read_timeout=read_timeout,phase_at_error=phase_at_error,
                          connection_started=connection_started,started=started,ended=ended,writes=conn.writes)
                (d/'http.json').write_bytes(catalog.canonical(http))
                # Exact data handed to transport, separately from observed bytes.
                (d/'wire.json').write_bytes(catalog.canonical({'headers':{'X-Request-Id':rid},'body':json.loads(body),
                    'prepared_body_sha256':catalog.digest(body),'outbound_body_sha256':outbound_body_sha256}))
                with self.lock:
                    self.connections.pop(rid,None)
                    if complete and status==200 and ended<self.end and not self.stop.is_set():
                        self.emit('response',rid=rid,status=status,response_ref=f'{rid}/body.bin',http_ref=f'{rid}/http.json')
                        self.pending.pop(rid)
                        success=True
                    elif status==200 and body_completed is not None and body_completed<self.end and not self.failed:
                        self.fail(rid,'completed HTTP body was not published before cutoff')
            return success

    def worker(self,index):
        self.ready.wait();job=self.job()
        while job:
            game,key,rid=job
            try:ok=self.once(key,rid,index)
            except Exception as exc:self.fail(rid,repr(exc));ok=False
            if not ok:return
            job=self.job(game)

    def run(self):
        for i in range(self.slots):
            thread=threading.Thread(target=self.worker,args=(i,),daemon=True);thread.start();self.threads.append(thread)
        with self.lock:
            self.origin=time.monotonic();self.end=self.origin+self.warmup+self.measure
            self.emit('replay_start',t=self.origin,**self.identity)
        self.ready.wait();measuring=False
        while time.monotonic()<self.end and not self.stop.is_set():
            now=time.monotonic()
            with self.lock:
                if not measuring and now>=self.origin+self.warmup:
                    self.emit('measure_start',boundary=self.origin+self.warmup);measuring=True
                self.emit('heartbeat')
            self.stop.wait(min(.2,max(0,self.end-now)))
        with self.lock:
            self.stop.set()
            # Boundary is declared independently of polling receipt timestamp.
            self.emit('measure_end',boundary=self.end)
            for rid in self.pending:self.emit('cutoff_pending',rid=rid)
        # On HTTP failure allow a bounded five-second error-body drain. No new
        # admission occurs once non-200 HEADERS are observed.
        drain=time.monotonic()+min(5.,max(0,float(os.environ.get('SCREEN_DEADLINE_MONOTONIC',str(time.monotonic()+5)))-time.monotonic()))
        while self.failed and any(t.is_alive() for t in self.threads) and time.monotonic()<drain:time.sleep(.01)
        with self.lock:
            for conn in self.connections.values():
                if time.monotonic()>=self.end:conn.cutoff_shutdown_at=time.monotonic()
                if conn.raw_socket:
                    try:conn.raw_socket.shutdown(socket.SHUT_RDWR)
                    except OSError:pass
                conn.close()
        for thread in self.threads:thread.join(timeout=2)
        with self.lock:
            self.emit('capture_end',threads_alive=sum(t.is_alive() for t in self.threads),failed=self.failed)
        self.log.close()
        return not self.failed and not any(t.is_alive() for t in self.threads)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=['prepare','run']);parser.add_argument('path');parser.add_argument('output');parser.add_argument('--arm',choices=['control','candidate']);args=parser.parse_args()
    if args.operation=='prepare':prepare(args.path,args.output);return 0
    identity={k:os.environ['SCREEN_'+k.upper()] for k in ('session_id','gpu_uuid','server_pgid','server_start')}
    return 0 if Replay(catalog.load(),args.path,args.output,'http://127.0.0.1:8001',args.arm,identity=identity).run() else 1

if __name__=='__main__':raise SystemExit(main())
