"""Small Linux x86-64 seccomp contract: query exec may create threads, no processes.

This is installed before executing any query. Failure/unknown architecture aborts.
Fork/vfork and non-CLONE_THREAD clone are denied; clone3 returns ENOSYS so libc may
use the inspectable clone ABI. The filter persists over exec and cannot be removed.
"""
import ctypes
import hashlib
import json
import os

ARCH=0xc000003e
ALLOW=0x7fff0000
ERRNO=0x00050000
PROGRAM=[(0x20,0,0,4),(0x15,1,0,ARCH),(0x06,0,0,0x80000000),
         (0x20,0,0,0),(0x45,0,1,0x40000000),(0x06,0,0,ERRNO|1),
         (0x15,0,1,57),(0x06,0,0,ERRNO|1),
         (0x15,0,1,58),(0x06,0,0,ERRNO|1),
         (0x15,0,1,435),(0x06,0,0,ERRNO|38),
         (0x15,0,4,56),(0x20,0,0,16),(0x45,0,1,0x10000),
         (0x06,0,0,ALLOW),(0x06,0,0,ERRNO|1),(0x06,0,0,ALLOW)]
SHA=hashlib.sha256(json.dumps(PROGRAM,separators=(',',':')).encode()).hexdigest()


class Filter(ctypes.Structure):
    _fields_=[('code',ctypes.c_ushort),('jt',ctypes.c_ubyte),('jf',ctypes.c_ubyte),('k',ctypes.c_uint)]


class Program(ctypes.Structure):
    _fields_=[('length',ctypes.c_ushort),('filters',ctypes.POINTER(Filter))]


def install():
    if os.uname().machine!='x86_64':raise RuntimeError('unsupported seccomp syscall architecture')
    libc=ctypes.CDLL(None,use_errno=True)
    rows=(Filter*len(PROGRAM))(*(Filter(*row) for row in PROGRAM));program=Program(len(rows),rows)
    if libc.prctl(38,1,0,0,0)!=0:raise OSError(ctypes.get_errno(),'PR_SET_NO_NEW_PRIVS')
    if libc.prctl(22,2,ctypes.byref(program),0,0)!=0:raise OSError(ctypes.get_errno(),'PR_SET_SECCOMP')
    if libc.prctl(39,0,0,0,0)!=1 or libc.prctl(21,0,0,0,0)!=2:raise RuntimeError('seccomp postcondition')
    fields=open('/proc/self/stat').read().rsplit(')',1)[1].split()
    return dict(contract='NO_NEW_PROCESSES_THREADS_ALLOWED',arch='x86_64',filter_sha256=SHA,
                no_new_privs=1,seccomp_mode=2,pid=os.getpid(),starttime=int(fields[19]))
