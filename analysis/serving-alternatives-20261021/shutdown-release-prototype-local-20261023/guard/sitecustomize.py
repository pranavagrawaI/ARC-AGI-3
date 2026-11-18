"""Invocation guard inherited by CPU fixture interpreters; not a GPU adapter."""
import os
import sys


def audit(event,args):
    if event in ('socket.connect','socket.connect_ex','socket.bind','socket.getaddrinfo'):
        raise RuntimeError('LOCAL PROTOTYPE: network forbidden')
    if event=='subprocess.Popen' and os.path.realpath(args[0])!=os.path.realpath(sys.executable):
        raise RuntimeError('LOCAL PROTOTYPE: only CPU Python fixtures may execute')
    if event=='open' and isinstance(args[0],str) and args[0].startswith('/dev/nvidia'):
        raise RuntimeError('LOCAL PROTOTYPE: GPU device forbidden')
    if event=='import' and args[0].split('.')[0] in ('torch','sglang','vllm','pynvml'):
        raise RuntimeError('LOCAL PROTOTYPE: native runtime/GPU import forbidden')


sys.addaudithook(audit)
