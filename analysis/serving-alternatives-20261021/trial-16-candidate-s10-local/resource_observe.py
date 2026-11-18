"""Candidate-only bounded five-probe receipt worker; no native/graceful proxy."""
import multiprocessing,os,time
import probe_ipc
from core import shutdown
PROBES=('gpu','apps','cgroup','port','disk')
before_deadline=shutdown.before_deadline
owned=shutdown.owned
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

