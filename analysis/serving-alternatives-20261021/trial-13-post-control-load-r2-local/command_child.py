"""Linux owned command containment. Parent policy chooses the executable."""
import ctypes
import json
import os
import signal
import sys
import no_processes

if __name__=='__main__':
    expected_parent=int(sys.argv[1]);argv=json.loads(sys.argv[2]);attestation_fd=int(sys.argv[3]);gate_fd=int(sys.argv[4])
    libc=ctypes.CDLL(None,use_errno=True)
    if libc.prctl(1,signal.SIGKILL,0,0,0)!=0:raise OSError(ctypes.get_errno(),'PR_SET_PDEATHSIG')
    if os.getppid()!=expected_parent:raise SystemExit(78)
    receipt=no_processes.install()
    os.write(attestation_fd,json.dumps(receipt).encode());os.close(attestation_fd)
    if os.read(gate_fd,1)!=b'G':raise SystemExit(79)
    os.close(gate_fd)
    if os.getppid()!=expected_parent:raise SystemExit(78)
    os.execv(argv[0],argv)
