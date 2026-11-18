"""Trusted sterile inspection entrypoint with an out-of-band startup seal.
Linux x86-64 only. All candidate execution follows seal(); no fallback.
Python starts before chroot; only explicit runtime files support later imports.
"""
import ctypes
import errno
import json
import os
import platform
import resource
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

LIBC = ctypes.CDLL(None, use_errno=True)
# x86-64 native ABI only; all unknown syscalls (including x32) return EPERM.
# No socket, process creation/exec, process inspection, signals, filesystem
# mutation, namespaces, privilege changes, io_uring, bpf or keyring calls.
ALLOWED = {0, 1, 2, 3, 5, 8, 9, 10, 11, 12, 13, 14, 15, 17, 19, 20,
           21, 24, 25, 28, 35, 39, 60, 63, 79, 89, 97, 102, 104,
           107, 108, 110, 115, 118, 120, 158, 186, 202, 218,
           228, 229, 230, 231, 257, 262, 267, 273, 318, 332}

class Filter(ctypes.Structure):
    _fields_ = [('code', ctypes.c_ushort), ('jt', ctypes.c_ubyte),
                ('jf', ctypes.c_ubyte), ('k', ctypes.c_uint)]
class Program(ctypes.Structure):
    _fields_ = [('length', ctypes.c_ushort), ('filters', ctypes.POINTER(Filter))]
class CapHeader(ctypes.Structure):
    _fields_ = [('version', ctypes.c_uint), ('pid', ctypes.c_int)]
class CapData(ctypes.Structure):
    _fields_ = [('effective', ctypes.c_uint), ('permitted', ctypes.c_uint),
                ('inheritable', ctypes.c_uint)]

def prctl(option, a=0, b=0, c=0, d=0):
    result = LIBC.prctl(ctypes.c_int(option), ctypes.c_ulong(a),
                        ctypes.c_ulong(b), ctypes.c_ulong(c), ctypes.c_ulong(d))
    if result < 0:
        raise OSError(ctypes.get_errno(), 'prctl %s' % option)
    return result

def qualify():
    if sys.platform != 'linux' or platform.machine() != 'x86_64' or ctypes.sizeof(ctypes.c_void_p) != 8:
        raise RuntimeError('UNSUPPORTED_PLATFORM')
    if os.geteuid() != 0:
        raise RuntimeError('ROOT_REQUIRED')
    if not hasattr(os, 'chroot') or not hasattr(os, 'setresuid'):
        raise RuntimeError('MISSING_IDENTITY_PRIMITIVE')

def seccomp():
    instructions = [(0x20, 0, 0, 4), (0x15, 1, 0, 0xC000003E),
                    (0x06, 0, 0, 0x80000000), (0x20, 0, 0, 0)]
    for number in sorted(ALLOWED):
        instructions += [(0x15, 0, 1, number), (0x06, 0, 0, 0x7FFF0000)]
    instructions += [(0x06, 0, 0, 0x00050000 | errno.EPERM)]
    filters = (Filter * len(instructions))(*(Filter(*x) for x in instructions))
    program = Program(len(instructions), filters)
    prctl(38, 1)
    if prctl(39) != 1:
        raise RuntimeError('NO_NEW_PRIVS_NOT_SET')
    if LIBC.prctl(22, 2, ctypes.byref(program), 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'seccomp install')

def arm_parent_death(parent_pid):
    prctl(1, signal.SIGKILL)
    observed = ctypes.c_int()
    if LIBC.prctl(2, ctypes.byref(observed), 0, 0, 0) != 0 or observed.value != signal.SIGKILL:
        raise RuntimeError('PDEATHSIG_NOT_SET')
    if os.getppid() != parent_pid:
        raise RuntimeError('PARENT_CHANGED')


def seal(root, parent_pid=None, receipt_fd=None):
    qualify()
    parent_pid = os.getppid() if parent_pid is None else parent_pid
    arm_parent_death(parent_pid)
    # This fresh exec has received no assessor data; pre-opened code/args only.
    # Close every extra descriptor, not merely those below a lowered rlimit.
    fds = [int(p.name) for p in Path('/proc/self/fd').iterdir()]
    null = os.open('/dev/null', os.O_RDONLY)
    os.dup2(null, 0)
    os.close(null)
    for fd in fds:
        if fd > 2 and fd != receipt_fd:
            try:
                os.close(fd)
            except OSError as error:
                if error.errno != errno.EBADF:
                    raise
    os.environ.clear()
    os.environ.update({'LANG': 'C', 'PYTHONDONTWRITEBYTECODE': '1'})
    os.chroot(root)
    os.chdir('/')
    sys.path[:] = ['/runtime']
    sys.argv[:] = ['candidate']
    # Drop ambient and EVERY supported bounding capability while privileged.
    prctl(47, 4)
    cap = 0
    while True:
        try:
            prctl(23, cap)
        except OSError as error:
            if error.errno == errno.EINVAL:
                break
            raise
        prctl(24, cap)
        if prctl(23, cap) != 0:
            raise RuntimeError('BOUNDING_CAP_REMAINS')
        if prctl(47, 1, cap) != 0:
            raise RuntimeError('AMBIENT_CAP_REMAINS')
        cap += 1
        if cap > 128:
            raise RuntimeError('UNKNOWN_CAP_RANGE')
    os.setgroups([])
    os.setresgid(65534, 65534, 65534)
    os.setresuid(65534, 65534, 65534)
    header = CapHeader(0x20080522, 0)
    data = (CapData * 2)()
    if LIBC.capset(ctypes.byref(header), data) != 0:
        raise OSError(ctypes.get_errno(), 'capset')
    if LIBC.capget(ctypes.byref(header), data) != 0 or any(x.effective or x.permitted or x.inheritable for x in data):
        raise RuntimeError('CAPS_REMAIN')
    if os.getresuid() != (65534,)*3 or os.getresgid() != (65534,)*3 or os.getgroups():
        raise RuntimeError('IDENTITY_REMAINS')
    # Identity transitions clear PDEATHSIG. Re-arm and verify the original
    # parent is still alive before the candidate can run.
    arm_parent_death(parent_pid)
    for name, value in [('RLIMIT_AS', 268435456), ('RLIMIT_CPU', 2),
                        ('RLIMIT_FSIZE', 65536), ('RLIMIT_CORE', 0),
                        ('RLIMIT_NOFILE', 32), ('RLIMIT_NPROC', 0)]:
        resource.setrlimit(getattr(resource, name), (value, value))
    receipt = {'uid': list(os.getresuid()), 'gid': list(os.getresgid()),
               'groups': os.getgroups(), 'capabilities_zero': True,
               'bounding_dropped': cap, 'ambient_cleared': True,
               'parent_death_signal': signal.SIGKILL,
               'cwd': os.getcwd(), 'no_new_privs': True, 'seccomp': 'native-allowlist'}
    seccomp()
    # Trusted startup receipt precedes the untrusted code. Assessor never treats
    # any later stdout as a trusted verdict; outputs require exact case matches.
    destination = 2 if receipt_fd is None else receipt_fd
    os.write(destination, (json.dumps({'sealed': receipt})+'\n').encode())
    if receipt_fd is not None:
        os.close(receipt_fd)

def main():
    if len(sys.argv) != 4 or not sys.argv[2].isdigit() or int(sys.argv[2]) <= 1 or not sys.argv[3].isdigit():
        raise RuntimeError('ROOT_PARENT_AND_RECEIPT_FD_REQUIRED')
    parent_pid = int(sys.argv[2])
    receipt_fd = int(sys.argv[3])
    arm_parent_death(parent_pid)
    root = Path(sys.argv[1]).resolve(strict=True)
    source = (root/'work/candidate.py').read_text()
    request = json.loads((root/'work/args.json').read_text())
    if set(request) != {'mode', 'args'} or request['mode'] not in ('candidate', 'inspection') or not isinstance(request['args'], list):
        raise RuntimeError('BAD_PROTOCOL')
    if request['mode'] != 'inspection':
        raise RuntimeError('INSPECTION_MODE_REQUIRED')
    seal(str(root), parent_pid, receipt_fd)
    code = compile(source, '/work/candidate.py', 'exec')
    # Inspection source runs as a normal script. Its stdout/stderr remain
    # untrusted process streams; the startup seal used a separate descriptor.
    scope = {'__name__': '__main__'}
    exec(code, scope, scope)

if __name__ == '__main__':
    main()
