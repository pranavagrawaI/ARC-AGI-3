"""Candidate process/cgroup bindings. No shipping-native or graceful-log proxy."""
import hashlib
import os
from pathlib import Path
import time
import runtime_scope
from core import shutdown


def bind(child, argv, run_id, sources, observer_spec):
    owner = shutdown.read_process(child.pid)
    expected = hashlib.sha256(b''.join(os.fsencode(s) + b'\0' for s in argv)).hexdigest()
    if (owner is None or child.poll() is not None or owner['ppid'] != os.getpid()
            or owner['identity']['pgid'] != child.pid or owner['identity']['sid'] != child.pid
            or owner['identity']['cmdline_sha256'] != expected):
        raise shutdown.UnsafeIdentity('candidate direct child/session/actual argv mismatch')
    members = shutdown.group(child.pid)
    for pid, member in members.items():
        cursor = pid
        seen = set()
        while cursor != child.pid:
            if cursor in seen or cursor not in members:
                raise shutdown.UnsafeIdentity('unknown candidate ancestry')
            seen.add(cursor)
            cursor = members[cursor]['ppid']
        if member['identity']['uid'] != os.getuid():
            raise shutdown.UnsafeIdentity('candidate UID drift')
    binding = dict(controller=shutdown.read_process(os.getpid())['identity'], pgid=child.pid,
        frontend=child.pid, members={pid: row['identity'] for pid, row in members.items()},
        source_files=dict(sources), bound_at=time.monotonic(), run_id=run_id)
    shutdown.check_sources(sources)
    observed = runtime_scope.observer()
    if observed != observer_spec['observer_scope'] or observed['cgroup'] != owner['identity']['cgroup']:
        raise shutdown.UnsafeIdentity('candidate/observer cgroup or namespace mismatch')
    for name, value in observed['namespaces'].items():
        if os.readlink(f'/proc/{child.pid}/ns/{name}') != value:
            raise shutdown.UnsafeIdentity('candidate namespace mismatch')
    paths, line = runtime_scope.derive(observed['cgroup'], Path('/proc/self/mountinfo').read_text())
    scope = dict(run_id=run_id, observer=observed, mount_record=line,
        paths={name: str(path) for name, path in paths.items()},
        file_keys={name: runtime_scope.key(path) for name, path in paths.items()},
        binding=binding, bound_at=time.monotonic())
    return binding, scope


def check(scope, binding, paths):
    controller = shutdown.read_process(binding['controller']['pid'])
    if controller is None or controller['identity'] != binding['controller']:
        raise shutdown.UnsafeIdentity('candidate resource controller identity')
    if os.getpid() != controller['pid'] and os.getppid() != controller['pid']:
        raise shutdown.UnsafeIdentity('resource worker must be controller child')
    observed = runtime_scope.observer()
    current, line = runtime_scope.derive(observed['cgroup'], Path('/proc/self/mountinfo').read_text())
    if (observed != scope['observer'] or line != scope['mount_record']
            or scope['paths'] != {name: str(path) for name, path in current.items()}
            or scope['paths'] != {name: str(path) for name, path in paths.items()}
            or scope['file_keys'] != {name: runtime_scope.key(path) for name, path in current.items()}):
        raise shutdown.UnsafeIdentity('candidate resource cgroup/mount/file drift')
    root = binding['members'].get(binding['pgid'], binding['members'].get(str(binding['pgid'])))
    if observed['cgroup'] != root['cgroup']:
        raise shutdown.UnsafeIdentity('candidate recorded cgroup mismatch')
    return dict(run_id=scope['run_id'], observer_pid=os.getpid(), cgroup=observed['cgroup'],
        namespaces=observed['namespaces'], input_scope='CANDIDATE_PROCESS_CGROUP_NO_NATIVE_PROXY')
