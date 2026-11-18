"""CPU-only stand-in stdout/stderr; never imports or executes a GPU runtime."""
import os
from pathlib import Path
import sys
import time

if __name__=='__main__':
    kind=sys.argv[1]
    if kind=='file':sys.stdout.buffer.write(Path(sys.argv[2]).read_bytes())
    elif kind=='big':sys.stdout.buffer.write(b'x'*(256*1024))
    elif kind=='slow':
        Path(sys.argv[2]).write_text(str(os.getpid()));print('partial',flush=True);time.sleep(20)
    elif kind=='error':print('fixture failure',file=sys.stderr);raise SystemExit(7)
    else:raise ValueError('unknown CPU fixture')
