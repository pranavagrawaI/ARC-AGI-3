"""Native process trace only. No frontend/image/prompt-render assertions are invented."""
import atexit
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time

class Capture:
    def __init__(self, path, session_id, gpu_uuid, heartbeat=True):
        if not session_id or not gpu_uuid:
            raise ValueError('session and observed GPU identity required')
        self.pid = os.getpid()
        self.boot_id = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        self.session_id, self.gpu_uuid = session_id, gpu_uuid
        self.stream = Path(path).open('x', encoding='utf-8')
        self.lock = threading.RLock()
        self.seq, self.requests = 0, {}
        self.closed, self.failure = False, None
        self.stop = threading.Event()
        self.emit('capture_start', clock='CLOCK_MONOTONIC', schema='native-target-v1')
        self.worker = None
        if heartbeat:
            self.worker = threading.Thread(target=self._heartbeat, daemon=True)
            self.worker.start()

    def emit(self, kind, **fields):
        with self.lock:
            if self.failure is not None:
                raise RuntimeError('native capture failed') from self.failure
            if self.closed or os.getpid() != self.pid:
                raise RuntimeError('closed or fork-inherited native capture')
            row = dict(seq=self.seq, t=time.clock_gettime(time.CLOCK_MONOTONIC),
                       pid=self.pid, boot_id=self.boot_id, session_id=self.session_id,
                       gpu_uuid=self.gpu_uuid, kind=kind, **fields)
            self.stream.write(json.dumps(row, separators=(',',':'))+'\n')
            self.stream.flush()
            self.seq += 1

    def _heartbeat(self):
        try:
            while not self.stop.wait(1):
                self.emit('heartbeat')
        except Exception as exc:
            self.failure = exc

    def accept(self, nid, prompt_ids):
        with self.lock:
            if not isinstance(nid,str) or not nid or nid in self.requests:
                raise ValueError('missing/duplicate engine request ID')
            ids = list(prompt_ids)
            if not ids or any(type(token) is not int for token in ids):
                raise ValueError('native token IDs required; prompt embeddings unsupported')
            self.requests[nid] = dict(count=0, first=False, finished=False)
            self.emit('native_accept', nid=nid, prompt_tokens=len(ids),
                      prompt_ids_sha256=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest())

    def observe(self, nid, count, finished, reason):
        with self.lock:
            state = self.requests[nid]
            if type(count) is not int or count < state['count']:
                raise ValueError('native output count regressed')
            if state['finished']:
                if count != state['count']:
                    raise ValueError('output after native finish')
                return
            if count and not state['first']:
                self.emit('target_first', nid=nid, output_tokens=count)
                state['first'] = True
            state['count'] = count
            if finished:
                self.emit('native_finish', nid=nid, output_tokens=count, reason=reason)
                state['finished'] = True

    def close(self):
        self.stop.set()
        if self.worker:
            self.worker.join(timeout=2)
        with self.lock:
            if not self.closed:
                self.emit('capture_end', unfinished=[nid for nid,s in self.requests.items() if not s['finished']])
                self.closed = True
                self.stream.close()

_CAPTURE = None

def capture():
    global _CAPTURE
    if _CAPTURE is None:
        directory = Path(os.environ['SERVING_NATIVE_DIR'])
        session = os.environ['SERVING_SESSION_ID']
        actual = subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'], text=True, timeout=5).strip().splitlines()
        if len(actual) != 1 or not actual[0].startswith('GPU-'):
            raise RuntimeError('one visible physical GPU required by capture contract')
        directory.mkdir(parents=True, exist_ok=True)
        _CAPTURE = Capture(directory/f'native-{os.getpid()}.jsonl', session, actual[0])
        atexit.register(_CAPTURE.close)
    return _CAPTURE

def sg_queue(req, retracted):
    trace = capture()
    if retracted:
        trace.emit('retract', nid=req.rid)
    else:
        trace.accept(req.rid, req.origin_input_ids)

def sg_output(req):
    reason = type(req.finished_reason).__name__ if req.finished() else 'running'
    if 'ABORT' in reason.upper() or 'ERROR' in reason.upper():
        capture().emit('abort', nid=req.rid, reason=reason)
    # finished_len trims speculative outputs at a stopping token.
    count = req.finished_len if req.finished_len is not None else len(req.output_ids)
    capture().observe(req.rid, count, req.finished(), reason)

def sg_abort(recv_req):
    capture().emit('abort_signal', nid_prefix=recv_req.rid, abort_all=recv_req.abort_all)

def vl_accept(req):
    capture().accept(req.request_id, req.prompt_token_ids)

def vl_output(req):
    # Call after target tokens were appended and stop-trimmed; draft placeholders
    # and chunked prefill produce no tokens here.
    capture().observe(req.request_id, len(req.output_token_ids), False, 'running')

def vl_finish(req):
    reason = req.status.name
    if 'ABORT' in reason or 'ERROR' in reason:
        capture().emit('abort', nid=req.request_id, reason=reason)
    capture().observe(req.request_id, len(req.output_token_ids), True, reason)

def vl_preempt(req):
    capture().emit('preempt', nid=req.request_id)
