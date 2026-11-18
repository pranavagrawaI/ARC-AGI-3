"""Isolated CPU replay of exact failed source; root/clock transposed as fixtures.

Never grants authority: filesystem writes confined locally, ALL subprocess/network
dispatch forbidden. Corrected entrypoint stops at the first hardware phase.
"""
import base64, datetime, hashlib, importlib, json, lzma, os, pathlib, sys, types

case_path = pathlib.Path(sys.argv[1]).resolve()
case = json.loads((case_path / 'case.json').read_text())
source = (case_path / 'bootstrap.py').read_text()
real_path = type(pathlib.Path())
remote_root = case['remote_root']
local_root = case_path / 'extracted'
foreign = case_path / 'foreign'
foreign.mkdir()
os.chdir(foreign)
sys.path[:] = [p for p in sys.path if p and p.startswith(sys.base_prefix) and 'site-packages' not in p]
if case['shadow']:
    (foreign / 'release_runner.py').write_text("from pathlib import Path\nPath('wrong-module-executed').write_text('shadow')\ndef main():return 0\n")
    sys.path.insert(0, str(foreign))
initial_path = list(sys.path)


def LocalPath(*parts):
    value = os.fspath(real_path(*parts))
    if value == remote_root or value.startswith(remote_root + '/'):
        value = str(local_root) + value[len(remote_root):]
    return real_path(value)


# The executable notebook bytes/assignment/config remain exact. This explicit
# environment fixture relocates its filesystem and replays its historical clock.
pathlib.Path = LocalPath
historical = datetime.datetime.fromisoformat(case['confirmed_utc']) + datetime.timedelta(seconds=1)


class HistoricalClock(datetime.datetime):
    @classmethod
    def now(cls, tz=None):
        return historical.astimezone(tz) if tz else historical.replace(tzinfo=None)


datetime.datetime = HistoricalClock
boundaries = []
attempts = []
real_import = importlib.import_module


def controlled_import(name, *args, **kwargs):
    module = real_import(name, *args, **kwargs)
    if name == 'release_runner' and hasattr(module, 'Runner'):
        def cpu_phase(self):
            boundaries.append(dict(phase='hardware_and_queries', module_file=module.__file__,
                config_cwd=self.config['cwd'], argv=list(sys.argv),
                source_sha256=hashlib.sha256(real_path(module.__file__).read_bytes()).hexdigest()))
            self.emit('CPU_ONLY_ENTRYPOINT_BOUNDARY')
            raise RuntimeError('CPU_ONLY_STOP_BEFORE_HARDWARE_QUERY_MODEL')
        module.Runner.hardware_and_queries = cpu_phase
    return module


importlib.import_module = controlled_import
if case['poison']:
    name = case['poison']
    poison = types.ModuleType(name)
    poison.__file__ = str(foreign / (name + '.py'))
    def wrong_main():
        (foreign / 'wrong-module-executed').write_text('preloaded')
        return 0
    poison.main = wrong_main
    sys.modules[name] = poison


def audit(event, args):
    if event in ('subprocess.Popen', 'os.system', 'socket.connect', 'socket.getaddrinfo', 'os.kill', 'os.killpg'):
        attempts.append(event)
        raise RuntimeError('CPU replay forbids ' + event)
    if event == 'open':
        filename, mode, flags = args
        if isinstance(filename, (str, bytes, os.PathLike)):
            value = os.fsdecode(filename)
            if value.startswith(('/dev/nvidia', '/dev/dri', '/proc/driver/nvidia')):
                attempts.append(value)
                raise RuntimeError('GPU access forbidden')
            writing = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
            if writing and not real_path(value).resolve().is_relative_to(case_path):
                raise RuntimeError('write outside local fixture')


sys.addaudithook(audit)
error = None
try:
    exec(compile(source, '<exact-armed-bootstrap-local-replay>', 'exec'), {'__name__': '__main__'})
except BaseException as failure:
    error = dict(type=type(failure).__name__, message=str(failure))
finally:
    report = dict(case=case['name'], bootstrap_sha256=hashlib.sha256(source.encode()).hexdigest(),
        context=dict(foreign_cwd=str(foreign), initial_sys_path=initial_path, mapped_root=str(local_root),
            historical_clock=historical.isoformat(), synthetic_cpu_fixture_not_remote_authority=True),
        error=error, boundaries=boundaries, forbidden_attempts=attempts,
        wrong_module_executed=(foreign / 'wrong-module-executed').exists(),
        extracted_files=sorted(str(p.relative_to(local_root)) for p in local_root.rglob('*') if p.is_file()),
        result=json.loads((local_root / 'output/result.json').read_text()) if (local_root / 'output/result.json').exists() else None)
    (case_path / 'RESULT.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))
