"""Unintegrated Linux CPU prototype; no GPU/network queries or model operations.

Call bind while the wrapper and native frontend are alive. Release accepts local
observation providers; none is supplied implicitly, and unknown evidence fails.
The caller must also impose the documented 600-second invocation timeout.
"""
import hashlib
import math
import multiprocessing
import os
from pathlib import Path
import signal
import time
import probe_ipc

PROBES = ('gpu', 'apps', 'cgroup', 'port', 'disk', 'native')


class UnsafeIdentity(ValueError):
    pass


class DeadlineReached(TimeoutError):
    pass


def read_process(pid):
    root = Path('/proc')/str(pid)
    try:
        fields = (root/'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        exe = root/'exe'
        binary = exe.stat()
        return dict(pid=pid, ppid=int(fields[1]), state=fields[0], identity=dict(
            pid=pid, starttime=int(fields[19]), pgid=int(fields[2]), sid=int(fields[3]),
            uid=root.stat().st_uid, exe=os.readlink(exe),
            executable_stat=[binary.st_dev, binary.st_ino, binary.st_size, binary.st_mtime_ns],
            cmdline_sha256=hashlib.sha256((root/'cmdline').read_bytes()).hexdigest(),
            pid_namespace=os.readlink(root/'ns/pid'), cgroup=(root/'cgroup').read_text(),
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip()))
    except (FileNotFoundError, ProcessLookupError) as error:
        # A process disappearing is different from a still-live identity whose
        # executable/cmdline cannot be established (e.g. an exec race).
        try:
            final_fields=(root/'stat').read_text().rsplit(')',1)[1].split()
        except (FileNotFoundError,ProcessLookupError):
            return None
        if final_fields[0]=='Z':
            return None
        raise UnsafeIdentity(f'live process identity incomplete: {pid}') from error
    except PermissionError as error:
        raise UnsafeIdentity(f'process visibility unavailable: {pid}') from error


def group(pgid):
    values = {}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            fields = (path/'stat').read_text().rsplit(')', 1)[1].split()
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError as error:
            raise UnsafeIdentity('group visibility unavailable: '+path.name) from error
        if int(fields[2]) == pgid:
            value = read_process(int(path.name))
            if value is not None:
                values[value['pid']] = value
    return values


def check_sources(sources):
    if not sources:
        raise UnsafeIdentity('explicit frontend/native source bindings required')
    for path, expected in sources.items():
        source = Path(path)
        try:
            actual=hashlib.sha256(source.read_bytes()).hexdigest()
        except OSError as error:
            raise UnsafeIdentity('frontend/native source unavailable: '+path) from error
        if not source.is_absolute() or actual != expected:
            raise UnsafeIdentity('frontend/native source drift: '+path)


def bind(root, frontend_pid, expected_argv, source_files):
    check_sources(source_files)
    owner = read_process(root.pid)
    if (root.poll() is not None or owner is None or owner['ppid'] != os.getpid()
            or owner['identity']['pgid'] != root.pid or owner['identity']['sid'] != root.pid):
        raise UnsafeIdentity('root must be this controller direct child in its new session')
    members = group(root.pid)
    frontend = members.get(frontend_pid)
    expected = hashlib.sha256(b''.join(os.fsencode(s)+b'\0' for s in expected_argv)).hexdigest()
    if (frontend is None or frontend['ppid'] != root.pid
            or frontend['identity']['cmdline_sha256'] != expected
            or frontend['identity']['exe'] != owner['identity']['exe']):
        raise UnsafeIdentity('frontend ancestry/executable/exact argv mismatch')
    for pid, member in members.items():
        cursor = pid
        seen = set()
        while cursor != root.pid:
            if cursor in seen or cursor not in members:
                raise UnsafeIdentity('unowned or ambiguous group member')
            seen.add(cursor)
            cursor = members[cursor]['ppid']
        if member['identity']['uid'] != os.getuid():
            raise UnsafeIdentity('owned process UID mismatch')
    return dict(controller=read_process(os.getpid())['identity'], pgid=root.pid,
                frontend=frontend_pid, members={pid:m['identity'] for pid,m in members.items()},
                source_files=dict(source_files), bound_at=time.monotonic())


def owned(binding):
    check_sources(binding['source_files'])
    if read_process(os.getpid())['identity'] != binding['controller']:
        raise UnsafeIdentity('different controller identity')
    current = group(binding['pgid'])
    if set(current)-set(binding['members']):
        raise UnsafeIdentity('unknown process appeared in owned group')
    living = {}
    for pid, expected in binding['members'].items():
        value = read_process(pid)
        if value is not None:
            if value['identity'] != expected:
                raise UnsafeIdentity(f'identity drift/reuse: {pid}')
            living[pid] = value['identity']
    return living


def before_deadline(end):
    if time.monotonic() >= end:
        raise DeadlineReached('release phase/absolute cleanup deadline')


def signal_owned(binding, targets, sig, events, end):
    """Verify the whole live target set first; signal pidfds, never a numeric PGID."""
    living = owned(binding)
    handles = {}
    try:
        before_deadline(end)
        for pid in targets:
            if pid not in living:
                continue
            try:
                handle = os.pidfd_open(pid)
            except ProcessLookupError:
                continue
            handles[pid] = handle
            value = read_process(pid)
            if value is not None and value['identity'] != binding['members'][pid]:
                raise UnsafeIdentity(f'identity changed during pidfd binding: {pid}')
        # Also reject unknown entrants observed while handles were acquired.
        owned(binding)
        for pid, handle in handles.items():
            before_deadline(end)
            started = time.monotonic()
            try:
                signal.pidfd_send_signal(handle, sig)
            except ProcessLookupError:
                continue
            events.append(dict(event='signal', pid=pid, signal=int(sig), start=started, end=time.monotonic()))
    finally:
        for handle in handles.values(): os.close(handle)


def probe_worker(probes, write_fd, read_fd, end):
    os.close(read_fd)
    try:
        for name in PROBES:
            started = time.monotonic()
            try:
                value = probes[name]()
                if not isinstance(value, dict) or type(value.get('eligible')) is not bool:
                    raise ValueError('complete typed eligibility receipt required')
                if set(value)&{'name','start','end','_ipc'}:
                    raise ValueError('reserved receipt envelope fields')
                record = dict(name=name, start=started, end=time.monotonic(), **value)
            except BaseException as error:
                record = dict(name=name, start=started, end=time.monotonic(), eligible=False, error=repr(error)[:512])
            try:
                probe_ipc.write_frame(write_fd,record,end)
            except probe_ipc.ReceiptError as error:
                # Never transfer an oversized/unsupported raw value. Large raw
                # artifacts require a separately bounded file/hash reference.
                record=dict(name=name,start=started,end=time.monotonic(),eligible=False,
                            error='receipt rejected: '+str(error)[:512])
                probe_ipc.write_frame(write_fd,record,end)
    except (BrokenPipeError,probe_ipc.TransferDeadline):
        pass
    finally:
        os.close(write_fd)


def observe(binding, probes, end, probe_seconds):
    before_deadline(end)
    started = time.monotonic()
    living = owned(binding)
    result = dict(start=started, process=dict(start=started, end=time.monotonic(),
                  eligible=not living, remaining=sorted(living)), probes={})
    context = multiprocessing.get_context('fork')
    read_fd,write_fd=os.pipe2(os.O_NONBLOCK|os.O_CLOEXEC)
    probe_end = min(end, time.monotonic()+probe_seconds)
    worker = context.Process(target=probe_worker, args=(probes,write_fd,read_fd,probe_end))
    result['ipc_deadline']=probe_end
    try:
        # Queries are synchronous controller-owned direct children. Receipt workers never launch them.
        if hasattr(probes,'prepare_controller'):
            result['controller_queries']=probes.prepare_controller(probe_end)
        before_deadline(probe_end)
        worker.start()
    except BaseException:
        os.close(read_fd);os.close(write_fd);worker.close()
        raise
    os.close(write_fd)
    result['ipc_error']=None
    try:
        while True:
            received=probe_ipc.read_frame(read_fd,probe_end)
            if received is None:
                if len(result['probes'])!=len(PROBES):raise probe_ipc.ReceiptError('incomplete receipt set')
                break
            record,transfer=received
            index=len(result['probes'])
            if (index>=len(PROBES) or not isinstance(record,dict) or record.get('name')!=PROBES[index]
                    or type(record.get('eligible')) is not bool
                    or any(type(record.get(k)) not in (int,float) for k in ('start','end'))
                    or not started<=record['start']<=record['end']<=transfer['end']):
                raise probe_ipc.ReceiptError('receipt identity/order/clock/schema mismatch')
            record['_ipc']=transfer
            result['probes'][record['name']] = record
    except (probe_ipc.ReceiptError,probe_ipc.TransferDeadline) as error:
        result['ipc_error']=repr(error)
    finally:
        os.close(read_fd)
        if worker.is_alive(): worker.kill()
        worker.join(timeout=max(0, min(.02, end-time.monotonic())))
        result['probe_worker_alive'] = worker.is_alive()
        if not worker.is_alive(): worker.close()
    for name in PROBES:
        result['probes'].setdefault(name, dict(name=name, start=None, end=None,
             eligible=False, error='not observed: probe/absolute deadline or worker failure'))
    result['end'] = time.monotonic()
    result['eligible'] = (result['end'] <= end and not result['probe_worker_alive']
                          and result['ipc_error'] is None
                          and result['process']['eligible']
                          and all(p['eligible'] for p in result['probes'].values()))
    return result


def release(binding, probes, *, end, graceful_seconds, fallback_reserve, poll=.02, probe_seconds=.08):
    started = time.monotonic()
    values = (end, graceful_seconds, fallback_reserve, poll, probe_seconds)
    if (not all(isinstance(x,(float,int)) and not isinstance(x,bool) and math.isfinite(x) for x in values)
            or not 0 < end-started <= 600 or min(values[1:]) <= 0
            or fallback_reserve >= end-started or set(probes) != set(PROBES)):
        raise ValueError('explicit finite <=600s deadline, cleanup reserve and exact probes required')
    events, samples = [], []
    fallback_used = False
    error = None
    passed = False
    try:
        signal_owned(binding, [binding['frontend']], signal.SIGTERM, events, end)
        graceful_end = min(started+graceful_seconds, end-fallback_reserve)
        while time.monotonic() < graceful_end:
            try:
                sample = observe(binding, probes, graceful_end, probe_seconds)
            except DeadlineReached:
                break # graceful budget expires into the existing cleanup reserve
            samples.append(sample)
            if sample['eligible']:
                passed = True
                break
            time.sleep(max(0, min(poll, graceful_end-time.monotonic())))
        if not passed:
            fallback_used = True
            # Preserve one end for TERM/KILL and every resource observation.
            signal_owned(binding, list(binding['members']), signal.SIGKILL, events, end)
            while time.monotonic() < end:
                sample = observe(binding, probes, end, probe_seconds)
                samples.append(sample)
                if sample['eligible']:
                    passed = True
                    break
                time.sleep(max(0,min(poll,end-time.monotonic())))
    except BaseException as failure:
        error = repr(failure)
    cleanup_error = None
    try: remaining = sorted(owned(binding))
    except UnsafeIdentity as failure:
        remaining = None
        cleanup_error = repr(failure)
    if remaining:
        cleanup_error = 'owned process cleanup incomplete at absolute deadline'
    finished = time.monotonic()
    ok = passed and not error and not cleanup_error and remaining == [] and finished <= end
    return dict(**{'pass':ok},
                start=started, end=finished, absolute_deadline=end, fallback_used=fallback_used,
                error=error, cleanup_error=cleanup_error, remaining=remaining, events=events, samples=samples,
                limitation='CPU prototype only; providers do not prove native GPU release')
