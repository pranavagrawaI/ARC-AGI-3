"""Reviewed trial10 phase utilities only; native release is controller-owned separately."""
import argparse
import csv
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.parse import urlsplit

GPU_ARGV=['nvidia-smi', '--query-gpu=name,memory.total,driver_version,uuid', '--format=csv,noheader,nounits']

def validate(config):
    from release_runner import validate
    return validate(config)

def require(condition, message):
    if not condition:
        raise ValueError(message)

def positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0

def group_members(pgid):
    """Linux group identity, including descendants after their leader exits; omit zombies."""
    result = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            fields = (p / 'stat').read_text().rsplit(')', 1)[1].split()
            if int(fields[2]) == pgid and fields[0] != 'Z':
                result.append(int(p.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    return sorted(result)

def startup_http_not_ready(code, path, allowed):
    """Only the pinned empty HTTP503 shape is eligible before first readiness."""
    if code != 1 or allowed != [503]:
        return False
    raw = Path(path).read_bytes()
    try:
        first, body = raw.split(b'\n', 1)
        metadata = json.loads(first)
        headers = metadata['headers']
        lengths = [v for k, v in headers if k.lower() == 'content-length']
        transfer = [v for k, v in headers if k.lower() == 'transfer-encoding']
        return (metadata['status'] == 503 and metadata['body_limit'] == 65536
                and metadata['body_format'] == 'raw bytes after first newline'
                and lengths == ['0'] and not transfer and body == b'')
    except (ValueError, KeyError, TypeError, AttributeError):
        return False

class PhaseTools:
    def __init__(self, config, output):
        self.config = validate(config)
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.started = time.monotonic()
        self.end = self.started + config['total_seconds']
        self.work_end = self.end - config['cleanup_seconds']
        self.groups = {}
        self.cleanup_failures = []
        self.handles = []
        self.phase_handles = {}
        self.sequence = 0
        self.trace = (self.output / 'supervisor.jsonl').open('x')
        self.emit('start', config_sha256=hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                  boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), pid=os.getpid(),
                  deadline=self.end, work_deadline=self.work_end, run_id=config['run_id'], purpose=config['purpose'],
                  python=sys.version, uname=list(os.uname()), total_seconds=config['total_seconds'])
        for name, path in [('meminfo', '/proc/meminfo'), ('cgroup_memory_max', '/sys/fs/cgroup/memory.max'),
                           ('cgroup_cpu_max', '/sys/fs/cgroup/cpu.max')]:
            p = Path(path)
            self.emit('resource', name=name, value=p.read_text() if p.exists() else None)

    def emit(self, event, **fields):
        self.trace.write(json.dumps(dict(seq=self.sequence, t=time.monotonic(), event=event, **fields)) + '\n')
        self.trace.flush()
        self.sequence += 1

    def deadline(self, seconds):
        end = min(self.work_end, time.monotonic() + seconds)
        require(end > time.monotonic(), 'overall work deadline reached')
        return end

    def start(self, name, spec, end):
        require(time.monotonic() < end, 'phase deadline reached before spawn')
        stdout = (self.output / f'{name}.stdout').open('xb')
        try:stderr = (self.output / f'{name}.stderr').open('xb')
        except BaseException:
            stdout.close()
            raise
        self.handles.extend((stdout, stderr))
        env = dict(os.environ, **spec.get('env', {}), SCREEN_PHASE=name,
                   SCREEN_DEADLINE_MONOTONIC=str(end), SCREEN_OVERALL_DEADLINE_MONOTONIC=str(self.end),
                   SCREEN_OUTPUT=str(self.output.resolve()), SCREEN_RUN_ID=self.config['run_id'])
        arm=name.split('.',1)[0]
        if arm in ('control','candidate'):
            package_output=self.output/(arm+'-package');package_output.mkdir(exist_ok=True)
            env['SCREEN_OUTPUT']=str(package_output.resolve())
        try:
            if name=='candidate.prepare':self.candidate_gate(end)
            p = subprocess.Popen(spec['argv'], cwd=self.config['cwd'], env=env, stdin=subprocess.DEVNULL,
                                 stdout=stdout, stderr=stderr, start_new_session=True, shell=False)
        except BaseException:
            for handle in (stdout,stderr):
                handle.close()
                self.handles.remove(handle)
            raise
        self.groups[p.pid] = p
        self.phase_handles[p.pid] = (stdout, stderr)
        self.emit('spawn', phase=name, pid=p.pid, pgid=p.pid, argv=spec['argv'], deadline=end)
        return p

    def wait(self, name, p, end, server=None):
        while True:
            if server is not None and server.poll() is not None:
                raise RuntimeError(f'{name}: server died with {server.returncode}')
            code = p.poll()
            if code is not None:
                self.emit('exit', phase=name, pid=p.pid, returncode=code)
                require(time.monotonic() <= end, f'{name}: late completion')
                require(code == 0, f'{name}: exit {code}')
                return
            require(time.monotonic() < end, f'{name}: absolute deadline')
            time.sleep(min(.02, max(0, end - time.monotonic())))

    def command(self, name, spec, server=None):
        end = self.deadline(spec['seconds'])
        p = self.start(name, spec, end)
        with self.cleanup_scope(name, [p.pid]):
            self.wait(name, p, end, server)

    def cleanup(self, groups=None):
        groups = list(self.groups) if groups is None else groups
        cleanup_end = min(self.end, time.monotonic() + self.config['cleanup_seconds'])
        for pgid in groups:
            before = group_members(pgid)
            if before:
                try:
                    os.killpg(pgid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        grace = min(cleanup_end, time.monotonic() + min(1, self.config['cleanup_seconds'] / 2))
        while any(group_members(g) for g in groups) and time.monotonic() < grace:
            time.sleep(.01)
        for pgid in groups:
            if group_members(pgid):
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        # SIGKILL delivery and descendant teardown are asynchronous. Observe the
        # actual group through one common bounded deadline, not just leader.wait().
        while any(group_members(g) for g in groups) and time.monotonic() < cleanup_end:
            time.sleep(min(.01, max(0, cleanup_end - time.monotonic())))
        for pgid in groups:
            p = self.groups[pgid]
            # No unbounded wait. Group kill is recorded even if kernel teardown is delayed.
            try:
                p.wait(timeout=max(0, min(.2, cleanup_end - time.monotonic())))
            except subprocess.TimeoutExpired:
                pass
            alive = group_members(pgid)
            self.emit('cleanup', pgid=pgid, remaining=alive, returncode=p.poll())
            if not alive:
                self.groups.pop(pgid, None)
                for handle in self.phase_handles.pop(pgid, ()):
                    handle.close()
                    self.handles.remove(handle)
        require(not any(group_members(g) for g in groups), 'process group cleanup incomplete; stop candidate')

    @contextmanager
    def cleanup_scope(self, context, groups=None):
        primary_error = None
        try:
            yield
        except BaseException as error:
            primary_error = error
            raise
        finally:
            try:
                self.cleanup(groups)
            except BaseException as cleanup_error:
                record = dict(context=context, error=repr(cleanup_error),
                              primary_error=repr(primary_error) if primary_error else None)
                self.cleanup_failures.append(record)
                self.emit('cleanup_failure', **record)
                if primary_error is None:
                    raise

    def hardware(self):
        self.command('hardware', self.config['hardware'])
        text = (self.output / 'hardware.stdout').read_text()
        rows = list(csv.reader(text.strip().splitlines()))
        require(len(rows) == 1 and len(rows[0]) == 4, 'exactly one complete GPU identity required')
        name, memory, driver, uuid = [s.strip() for s in rows[0]]
        require(name=='NVIDIA RTX PRO 6000 Blackwell Server Edition' and int(memory) >= self.config['hardware']['minimum_memory_mib'], 'wrong hardware; no preparation allowed')
        require(driver and uuid.startswith('GPU-'), 'GPU driver/UUID missing')
        self.emit('hardware_pass', name=name, memory_mib=int(memory), driver=driver, uuid=uuid)
        self.gpu_uuid=uuid

    def ready(self, arm, a, server, server_end):
        end = min(server_end, self.deadline(a['ready_seconds']))
        n = 0
        while True:
            require(server.poll() is None, f'{arm}: server died before readiness')
            request_end = min(end, time.monotonic() + a['request_seconds'])
            spec = dict(argv=[sys.executable, str(Path(__file__).with_name('http_once.py')), a['ready_url']], seconds=a['request_seconds'])
            name = f'{arm}.health-{n}'
            p = self.start(name, spec, request_end)
            with self.cleanup_scope(name, [p.pid]):
                while p.poll() is None:
                    require(server.poll() is None, f'{arm}: server died during readiness')
                    require(time.monotonic() < request_end, f'{arm}: health absolute deadline')
                    time.sleep(.01)
                require(time.monotonic() <= request_end, f'{arm}: late health response')
                code = p.returncode
                self.emit('exit', phase=name, pid=p.pid, returncode=code)
            if code == 0:
                self.emit('ready', arm=arm)
                return
            # HTTP503 is NOT ready or identified transient. Only explicit pinned
            # empty503 policy may keep waiting inside the ORIGINAL startup end.
            not_ready = startup_http_not_ready(code, self.output / (name + '.stdout'),
                                               a.get('ready_retry_http_statuses', []))
            require(code == 75 or not_ready, f'{arm}: health HTTP/error status {code}')
            if not_ready:
                self.emit('startup_not_ready', arm=arm, phase=name, status=503, deadline=end)
            require(time.monotonic() < end, f'{arm}: readiness deadline')
            n += 1
            time.sleep(min(a.get('ready_poll_seconds', 2), max(0, end - time.monotonic())))

