"""Linux owned command containment. Parent policy chooses the executable."""
import ctypes
import json
import os
import signal
import sys

if __name__=='__main__':
    expected_parent=int(sys.argv[1]);argv=json.loads(sys.argv[2])
    libc=ctypes.CDLL(None,use_errno=True)
    if libc.prctl(1,signal.SIGKILL,0,0,0)!=0:raise OSError(ctypes.get_errno(),'PR_SET_PDEATHSIG')
    if os.getppid()!=expected_parent:raise SystemExit(78)
    os.execv(argv[0],argv)
