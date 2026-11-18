"""CPU stand-in for the same prepare/serve registration call, no model setup."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import registration

if __name__=='__main__':
    prepared,source_map,output,run_id=sys.argv[1:]
    data=json.loads(Path(prepared).read_text())
    child=subprocess.Popen(data['argv'],stdin=subprocess.DEVNULL,start_new_session=False)
    registration.emit(child.pid,prepared,json.loads(Path(source_map).read_text()),run_id,output,time.monotonic()+2)
    raise SystemExit(child.wait())
