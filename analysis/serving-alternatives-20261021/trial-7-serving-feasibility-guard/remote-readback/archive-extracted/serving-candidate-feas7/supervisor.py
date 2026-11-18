"""One-shot local process supervisor. No SDK, uploader, installer or model defaults."""
import argparse
import csv
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

GPU_ARGV = ['nvidia-smi', '--query-gpu=name,memory.total,driver_version,uuid',
            '--format=csv,noheader,nounits']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def validate(config, check_cwd=True):
    require(config['mode'] in ('GPU_SCREEN', 'CPU_FIXTURE_ONLY'), 'unknown mode')
    require(isinstance(config['run_id'], str) and config['run_id'], 'run id missing')
    require(config['purpose'] in ('PAIRED_CAPACITY', 'CANDIDATE_FEASIBILITY'), 'explicit scientific purpose required')
    require(positive(config['total_seconds']), 'explicit finite total cap required')
    require(positive(config['cleanup_seconds']) and config['cleanup_seconds'] < config['total_seconds'], 'cleanup reserve required')
    require(positive(config['hardware']['seconds']), 'hardware ceiling missing')
    require(config['hardware']['minimum_memory_mib'] >= 90000, 'memory guard weakened')
    if config['mode'] == 'GPU_SCREEN':
        require(config['hardware']['argv'] == GPU_ARGV, 'GPU identity must use actual nvidia-smi')
    specs = [config['hardware']]
    for arm in arm_order(config):
        a = config[arm]
        specs.extend(a[p] for p in ('prepare', 'identity', 'server', work_phase(config), 'qualify'))
        for p in ('ready_seconds', 'request_seconds', 'drain_seconds'):
            require(positive(a[p]), f'{arm}.{p} missing')
        poll = a.get('ready_poll_seconds', 2)
        require(positive(poll) and (config['mode'] != 'GPU_SCREEN' or poll >= 2), 'GPU readiness cadence must be at least 2 seconds')
        u = urlsplit(a['ready_url'])
        require(u.scheme == 'http' and u.hostname in ('127.0.0.1', 'localhost') and not u.username, 'health must be local HTTP')
        require(u.port is not None, 'explicit local server port required')
    for spec in specs:
        argv = spec['argv']
        require(isinstance(argv, list) and argv and all(isinstance(s, str) and s and '\0' not in s for s in argv), 'argv array required')
        require(positive(spec['seconds']), 'explicit phase ceiling required')
        require(isinstance(spec.get('env', {}), dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in spec.get('env', {}).items()), 'environment must be string map')
    require(isinstance(config['cwd'], str) and Path(config['cwd']).is_absolute(), 'absolute cwd required')
    if check_cwd:
        require(Path(config['cwd']).is_dir(), 'working directory missing')
    return config


def arm_order(config):
    expected = ['candidate'] if config['purpose'] == 'CANDIDATE_FEASIBILITY' else ['control', 'candidate']
    require(config.get('arm_order', expected) == expected, 'arm order disagrees with purpose')
    return expected


def work_phase(config):
    return 'functional' if config['purpose'] == 'CANDIDATE_FEASIBILITY' else 'replay'


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


class Supervisor:
    def __init__(self, config, output):
        self.config = validate(config)
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.started = time.monotonic()
        self.end = self.started + config['total_seconds']
        self.work_end = self.end - config['cleanup_seconds']
        self.groups = {}
        self.handles = []
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
        stderr = (self.output / f'{name}.stderr').open('xb')
        self.handles.extend((stdout, stderr))
        env = dict(os.environ, **spec.get('env', {}), SCREEN_PHASE=name,
                   SCREEN_DEADLINE_MONOTONIC=str(end), SCREEN_OVERALL_DEADLINE_MONOTONIC=str(self.end),
                   SCREEN_OUTPUT=str(self.output.resolve()), SCREEN_RUN_ID=self.config['run_id'])
        p = subprocess.Popen(spec['argv'], cwd=self.config['cwd'], env=env, stdin=subprocess.DEVNULL,
                             stdout=stdout, stderr=stderr, start_new_session=True, shell=False)
        self.groups[p.pid] = p
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
        try:
            self.wait(name, p, end, server)
        finally:
            self.cleanup([p.pid])

    def cleanup(self, groups=None):
        groups = list(self.groups) if groups is None else groups
        for pgid in groups:
            before = group_members(pgid)
            if before:
                try:
                    os.killpg(pgid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        grace = min(self.end, time.monotonic() + min(1, self.config['cleanup_seconds'] / 2))
        while any(group_members(g) for g in groups) and time.monotonic() < grace:
            time.sleep(.01)
        for pgid in groups:
            if group_members(pgid):
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            p = self.groups[pgid]
            # No unbounded wait. Group kill is recorded even if kernel teardown is delayed.
            try:
                p.wait(timeout=max(0, min(.2, self.end - time.monotonic())))
            except subprocess.TimeoutExpired:
                pass
            alive = group_members(pgid)
            self.emit('cleanup', pgid=pgid, remaining=alive, returncode=p.poll())
            if not alive:
                self.groups.pop(pgid, None)
        require(not any(group_members(g) for g in groups), 'process group cleanup incomplete; stop candidate')

    def hardware(self):
        self.command('hardware', self.config['hardware'])
        text = (self.output / 'hardware.stdout').read_text()
        rows = list(csv.reader(text.strip().splitlines()))
        require(len(rows) == 1 and len(rows[0]) == 4, 'exactly one complete GPU identity required')
        name, memory, driver, uuid = [s.strip() for s in rows[0]]
        require('rtx pro 6000' in name.lower() and int(memory) >= self.config['hardware']['minimum_memory_mib'], 'wrong hardware; no preparation allowed')
        require(driver and uuid.startswith('GPU-'), 'GPU driver/UUID missing')
        self.emit('hardware_pass', name=name, memory_mib=int(memory), driver=driver, uuid=uuid)

    def ready(self, arm, a, server, server_end):
        end = min(server_end, self.deadline(a['ready_seconds']))
        n = 0
        while True:
            require(server.poll() is None, f'{arm}: server died before readiness')
            request_end = min(end, time.monotonic() + a['request_seconds'])
            spec = dict(argv=[sys.executable, str(Path(__file__).with_name('http_once.py')), a['ready_url']], seconds=a['request_seconds'])
            name = f'{arm}.health-{n}'
            p = self.start(name, spec, request_end)
            try:
                while p.poll() is None:
                    require(server.poll() is None, f'{arm}: server died during readiness')
                    require(time.monotonic() < request_end, f'{arm}: health absolute deadline')
                    time.sleep(.01)
                require(time.monotonic() <= request_end, f'{arm}: late health response')
                code = p.returncode
                self.emit('exit', phase=name, pid=p.pid, returncode=code)
            finally:
                self.cleanup([p.pid])
            if code == 0:
                self.emit('ready', arm=arm)
                return
            # Only connection refusal while a newly started server binds is retryable.
            require(code == 75, f'{arm}: health HTTP/error status {code}')
            require(time.monotonic() < end, f'{arm}: readiness deadline')
            n += 1
            time.sleep(min(a.get('ready_poll_seconds', 2), max(0, end - time.monotonic())))

    def arm(self, arm):
        a = self.config[arm]
        for phase in ('prepare', 'identity'):
            self.command(f'{arm}.{phase}', a[phase])
        # Prevent an unrelated pre-existing listener from satisfying readiness.
        with socket.socket() as check:
            check.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            check.bind(('127.0.0.1', urlsplit(a['ready_url']).port))
        startup_end = self.deadline(a['server']['seconds'])
        server = self.start(f'{arm}.server', a['server'], self.work_end)
        self.emit('startup_deadline', arm=arm, deadline=startup_end)
        try:
            self.ready(arm, a, server, startup_end)
            for phase in (work_phase(self.config), 'qualify'):
                spec = dict(a[phase])
                if phase in ('replay', 'functional'):
                    spec['env'] = dict(spec.get('env', {}), SCREEN_DRAIN_SECONDS=str(a['drain_seconds']))
                self.command(f'{arm}.{phase}', spec, server)
            self.emit('arm_pass', arm=arm)
        finally:
            self.cleanup()

    def run(self):
        result = dict(run_id=self.config['run_id'], mode=self.config['mode'], pass_=False)
        try:
            self.hardware()
            for arm in arm_order(self.config):
                self.arm(arm)
            result['pass_'] = True
        except BaseException as exc:
            result['error'] = f'{type(exc).__name__}: {exc}'
        finally:
            try:
                self.cleanup()
            except BaseException as exc:
                result['pass_'] = False
                result['cleanup_error'] = repr(exc)
            result.update(elapsed_seconds=time.monotonic() - self.started,
                          hard_cap_seconds=self.config['total_seconds'], surviving_groups=list(self.groups))
            self.emit('terminal', **result)
            (self.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
            for h in self.handles:
                h.close()
            self.trace.close()
        return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('config')
    p.add_argument('output')
    p.add_argument('--cpu-fixture', action='store_true')
    p.add_argument('--assignment')
    args = p.parse_args()
    config = validate(json.loads(Path(args.config).read_text()))
    if args.cpu_fixture:
        require(config['mode'] == 'CPU_FIXTURE_ONLY', 'fixture flag cannot execute GPU mode')
    else:
        require(config['mode'] == 'GPU_SCREEN', 'CPU fixture requires explicit flag')
        assignment = json.loads(Path(args.assignment).read_text())
        require(assignment['launch_authorized'] is True and assignment['serialized_assignment'], 'reviewed serialized assignment required')
        sha = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        require(assignment['config_sha256'] == sha and assignment['total_seconds'] == config['total_seconds'], 'assignment/config/cap mismatch')
    def stop(signum, frame):
        raise RuntimeError(f'signal {signum}')
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    result = Supervisor(config, args.output).run()
    print(json.dumps(result), flush=True)
    return 0 if result['pass_'] else 1


if __name__ == '__main__':
    sys.exit(main())
