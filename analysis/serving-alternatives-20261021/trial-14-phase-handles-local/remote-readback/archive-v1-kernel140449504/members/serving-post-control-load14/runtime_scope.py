"""Release-only runtime boundaries: intended cgroup paths and fresh native stderr.

No GPU commands. Unsupported/ambiguous scope is an error. Saved CPU fixtures are
not accepted as live paths. Bind while the exact registered frontend is alive.
"""
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import re
import stat
import time
import uuid

from core import shutdown


def small(path):
    if Path(path).stat().st_size>16384:raise ValueError('runtime binding receipt cap')
    with Path(path).open('rb') as stream:raw=stream.read(16385)
    if len(raw)>16384:raise ValueError('runtime binding receipt cap')
    return raw
def digest(path):return hashlib.sha256(small(path)).hexdigest()
def key(path):
    value=Path(path).stat()
    return [value.st_dev,value.st_ino,stat.S_IFMT(value.st_mode)]
def exact_path(path):
    path=Path(path)
    if not path.is_absolute() or path.resolve()!=path:raise ValueError('absolute nonsymlink runtime path required')
    return path

def unescape(value):return re.sub(r'\\([0-7]{3})',lambda m:chr(int(m[1],8)),value)


def derive(cgroup,mountinfo):
    rows=[r[3:] for r in cgroup.splitlines() if r.startswith('0::')]
    if len(rows)!=1:raise ValueError('exact one cgroup2 hierarchy required')
    group=PurePosixPath(rows[0]);matches=[]
    if not group.is_absolute() or '..' in group.parts:raise ValueError('cgroup hierarchy path')
    for line in mountinfo.splitlines():
        before,sep,after=line.partition(' - ')
        tail=after.split()
        if not sep or not tail or tail[0]!='cgroup2':continue
        fields=before.split()
        if len(fields)<6:raise ValueError('malformed cgroup2 mount')
        root=PurePosixPath(unescape(fields[3]));mount=PurePosixPath(unescape(fields[4]))
        if not root.is_absolute() or not mount.is_absolute() or '..' in root.parts+mount.parts:raise ValueError('cgroup2 mount path')
        if group.is_relative_to(root):matches.append((line,Path(str(mount/group.relative_to(root)))))
    if len(matches)!=1:raise ValueError('ambiguous or unavailable intended cgroup2 mount')
    line,directory=matches[0]
    return dict(meminfo=Path('/proc/meminfo'),maximum=directory/'memory.max',current=directory/'memory.current',
                stat=directory/'memory.stat',disk=Path('/tmp')),line


def observer():
    return dict(cgroup=Path('/proc/self/cgroup').read_text(),
        namespaces={name:os.readlink('/proc/self/ns/'+name) for name in ('pid','mnt','cgroup')})


def create_native_output(parent,run_id):
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,100}',run_id):raise ValueError('bounded run identity')
    parent=exact_path(parent);directory=parent/run_id;directory.mkdir(mode=0o700,exist_ok=False)
    log=directory/'native.stderr'
    fd=os.open(log,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600);os.close(fd)
    created=dict(run_id=run_id,directory=str(directory),directory_key=key(directory),log_path=str(log),log_key=key(log),
        controller=shutdown.read_process(os.getpid())['identity'],nonce=uuid.uuid4().hex,created_at=time.monotonic())
    marker=directory/'RUN-IDENTITY.json'
    with marker.open('x') as stream:json.dump(created,stream,sort_keys=True)
    return dict(created=created,marker_path=str(marker),marker_sha256=digest(marker))


def check_native_output(lease):
    created=lease['created'];directory=exact_path(created['directory']);log=exact_path(created['log_path'])
    marker=exact_path(lease['marker_path'])
    if (log.parent!=directory or marker!=directory/'RUN-IDENTITY.json' or key(directory)!=created['directory_key']
            or key(log)!=created['log_key'] or created['log_key'][2]!=stat.S_IFREG
            or digest(marker)!=lease['marker_sha256'] or json.loads(small(marker))!=created):
        raise ValueError('fresh native output identity/path drift')
    return log


def open_native_stderr(lease):
    """Runner opens this before Popen; it retains Popen ownership on registration errors."""
    log=check_native_output(lease);created=lease['created']
    if shutdown.read_process(os.getpid())['identity']!=created['controller']:raise ValueError('native output controller drift')
    fd=os.open(log,os.O_WRONLY|os.O_APPEND|os.O_NOFOLLOW)
    try:
        value=os.fstat(fd)
        if [value.st_dev,value.st_ino,stat.S_IFMT(value.st_mode)]!=created['log_key'] or value.st_size!=0:
            raise ValueError('native log must be fresh and empty before wrapper launch')
        return os.fdopen(fd,'ab',buffering=0)
    except BaseException:
        os.close(fd);raise


def record_wrapper(child,lease,started):
    """Caller already retains this direct Popen child; this function never spawns."""
    check_native_output(lease);created=lease['created'];value=shutdown.read_process(child.pid)
    if (value is None or child.poll() is not None or value['ppid']!=os.getpid()
            or value['identity']['pgid']!=child.pid or value['identity']['sid']!=child.pid
            or shutdown.read_process(os.getpid())['identity']!=created['controller']):
        raise ValueError('native log wrapper ownership/identity unavailable')
    launch=dict(wrapper=value['identity'],marker_sha256=lease['marker_sha256'],start=started,end=time.monotonic())
    path=Path(created['directory'])/'NATIVE-LAUNCH.json'
    with path.open('x') as stream:json.dump(launch,stream,sort_keys=True)
    lease.update(launch_path=str(path),launch_sha256=digest(path))
    return lease


def native_binding(binding,lease,run_id):
    log=check_native_output(lease);created=lease['created']
    launch_path=exact_path(lease['launch_path'])
    if launch_path!=Path(created['directory'])/'NATIVE-LAUNCH.json' or digest(launch_path)!=lease['launch_sha256']:
        raise ValueError('native launch receipt drift')
    launch=json.loads(small(launch_path))
    if (created['run_id']!=run_id or binding.get('run_id')!=run_id or created['controller']!=binding['controller']
            or launch['wrapper']!=binding['members'][binding['pgid']]
            or launch['marker_sha256']!=lease['marker_sha256'] or not created['created_at']<=launch['start']<=launch['end']<=binding['bound_at']):
        raise ValueError('native output run/controller/wrapper binding mismatch')
    return log


def check_frontend(binding,require_alive=False):
    value=shutdown.read_process(binding['frontend'])
    if value is None:
        if require_alive:raise ValueError('scope binding requires live frontend')
        return False
    if value['identity']!=binding['members'][binding['frontend']]:raise ValueError('runtime frontend identity drift')
    return True


def bind(binding,paths,lease,run_id):
    check_frontend(binding,True);before=observer();frontend=binding['members'][binding['frontend']]
    if (before['cgroup']!=frontend['cgroup'] or any(os.readlink(f"/proc/{binding['frontend']}/ns/{name}")!=value
            for name,value in before['namespaces'].items())):raise ValueError('frontend/observer scope mismatch')
    intended,line=derive(before['cgroup'],Path('/proc/self/mountinfo').read_text())
    if {k:str(v) for k,v in paths.items()}!={k:str(v) for k,v in intended.items()}:raise ValueError('cgroup/disk paths not derived from intended frontend mount')
    keys={name:key(exact_path(path)) for name,path in intended.items()}
    log=native_binding(binding,lease,run_id)
    if (key(f"/proc/{binding['frontend']}/fd/2")!=lease['created']['log_key']
            or os.readlink(f"/proc/{binding['frontend']}/fd/2")!=str(log)):
        raise ValueError('frontend stderr is not fresh run native log')
    if observer()!=before:raise ValueError('observer changed during scope binding')
    return dict(run_id=run_id,observer=before,mount_record=line,paths={k:str(v) for k,v in intended.items()},file_keys=keys,
                lease=lease,bound_at=time.monotonic())


def check(scope,binding,paths,native_path):
    controller=shutdown.read_process(binding['controller']['pid'])
    if controller is None or controller['identity']!=binding['controller']:raise ValueError('runtime controller identity unavailable/drift')
    if os.getpid()!=controller['pid'] and os.getppid()!=controller['pid']:raise ValueError('observer must be controller or direct receipt worker')
    seen=observer()
    if seen!=scope['observer'] or seen['cgroup']!=binding['members'][binding['frontend']]['cgroup']:
        raise ValueError('observer/front-end cgroup or namespace drift')
    intended,line=derive(seen['cgroup'],Path('/proc/self/mountinfo').read_text())
    if line!=scope['mount_record'] or scope['paths']!={k:str(v) for k,v in intended.items()} or scope['paths']!={k:str(v) for k,v in paths.items()}:
        raise ValueError('intended cgroup mount/path drift')
    if {name:key(exact_path(path)) for name,path in intended.items()}!=scope['file_keys']:raise ValueError('cgroup/disk file identity drift')
    log=native_binding(binding,scope['lease'],scope['run_id'])
    if str(native_path)!=str(log):raise ValueError('stale/copied native path is not bound fresh output')
    if check_frontend(binding) and (key(f"/proc/{binding['frontend']}/fd/2")!=scope['lease']['created']['log_key']
            or os.readlink(f"/proc/{binding['frontend']}/fd/2")!=str(log)):
        raise ValueError('live frontend stderr drift')
    return dict(input_scope='LIVE_BOUND_CGROUP_AND_FRESH_STDERR',observer_pid=os.getpid(),run_id=scope['run_id'],
                cgroup=seen['cgroup'],namespaces=seen['namespaces'],mount_sha256=hashlib.sha256(line.encode()).hexdigest(),
                native_marker_sha256=scope['lease']['marker_sha256'])
