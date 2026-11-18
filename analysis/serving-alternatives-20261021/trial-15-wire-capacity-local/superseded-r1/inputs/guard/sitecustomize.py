"""CPU Python fixtures and IPv4 loopback only; no external network/GPU."""
import hashlib
import os
from pathlib import Path
import sys


def allowed_program(path):
    if os.path.realpath(path)==os.path.realpath(sys.executable):return True
    expected=os.environ.get('LOCAL_CPU_ENTRY')
    return (expected==str(path) and hashlib.sha256(Path(path).read_bytes()).hexdigest()==os.environ.get('LOCAL_CPU_ENTRY_SHA'))


def audit(event,args):
    if event in ('socket.connect','socket.connect_ex'):
        address=args[1]
        if not isinstance(address,tuple) or address[0]!='127.0.0.1':raise RuntimeError('only IPv4 loopback connections allowed')
    if event=='socket.getaddrinfo' and args[0]!='127.0.0.1':raise RuntimeError('external DNS forbidden')
    if event=='socket.bind':
        address=args[1]
        if not isinstance(address,tuple) or address[0]!='127.0.0.1':raise RuntimeError('only IPv4 loopback bind allowed')
    if event=='subprocess.Popen' and not allowed_program(args[0]):raise RuntimeError('only pinned CPU Python fixture allowed')
    if event=='os.exec' and os.path.realpath(args[0])!=os.path.realpath(sys.executable):raise RuntimeError('only CPU Python exec allowed')
    if event=='open' and isinstance(args[0],str) and args[0].startswith('/dev/nvidia'):raise RuntimeError('GPU access forbidden')
    if event=='import' and args[0].split('.')[0] in ('torch','sglang','vllm','pynvml'):raise RuntimeError('native runtime import forbidden')


sys.addaudithook(audit)
