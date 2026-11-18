"""Pure classification of saved observations; no GPU, socket or network calls.

Original numeric eligibility predicates are retained. Each receipt is independent;
port eligibility must be supplied by an explicit observation, never inferred.
"""
import csv
import io

GPU_MAX_MIB = 2048
HOST_MIN_BYTES = 140*1024**3
DISK_MIN_BYTES = 100*1024**3
GPU_NAME = 'NVIDIA RTX PRO 6000 Blackwell Server Edition'


def gpu(raw, expected_uuid):
    rows = list(csv.reader(io.StringIO(raw)))
    if len(rows)!=1 or len(rows[0])!=3: raise ValueError('one complete GPU release query')
    name, memory, uuid = [v.strip() for v in rows[0]]
    if name!=GPU_NAME or not expected_uuid.startswith('GPU-') or uuid!=expected_uuid:
        raise ValueError('bound release GPU identity')
    if not memory.isdigit(): raise ValueError('numeric GPU memory required')
    return dict(eligible=int(memory)<=GPU_MAX_MIB, used_mib=int(memory), threshold_mib=GPU_MAX_MIB, raw=raw)


def cgroup(raw):
    mem={k:int(v.split()[0])*1024 for k,v in (line.split(':',1) for line in raw['meminfo'].splitlines())}
    fields={k:int(v) for k,v in (line.split() for line in raw['stat'].splitlines())}
    available=mem['MemAvailable']
    if raw['maximum'].isdigit():
        available=min(available,max(0,int(raw['maximum'])-int(raw['current'])+fields.get('inactive_file',0)))
    return dict(eligible=available>=HOST_MIN_BYTES, computed_bytes=available, threshold_bytes=HOST_MIN_BYTES,
                accounting='unchanged max-current+inactive_file capped by MemAvailable', raw=raw)


def providers(raw, expected_uuid, port, native):
    def classify(fn,evidence):
        try:value=fn()
        except Exception as error:value=dict(eligible=False,error=repr(error),raw=evidence)
        return dict(**value,origin='SAVED_OR_SYNTHETIC_INPUT_NOT_LIVE_QUERY')
    return dict(gpu=lambda:classify(lambda:gpu(raw['gpu'],expected_uuid),raw['gpu']),
                apps=lambda:classify(lambda:dict(eligible=not raw['apps'].strip(),raw=raw['apps']),raw['apps']),
                cgroup=lambda:classify(lambda:cgroup({k:raw[k] for k in ('meminfo','maximum','current','stat')}),raw),
                disk=lambda:classify(lambda:dict(eligible=raw['disk']>=DISK_MIN_BYTES,free_bytes=raw['disk'],threshold_bytes=DISK_MIN_BYTES),raw['disk']),
                port=port,native=native)
