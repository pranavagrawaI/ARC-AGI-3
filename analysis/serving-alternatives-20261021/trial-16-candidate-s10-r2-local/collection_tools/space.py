"""Local simultaneous storage admission; no network/GPU/credentials."""
import os,shutil,time
from pathlib import Path
ARCHIVE=2*1024**3
EXTRACTED=8*1024**3
DERIVED=2*1024**3
RESERVE=1024**3

def check(path,stage='archive'):
    path=Path(path);start=time.monotonic();usage=shutil.disk_usage(path)
    required=(ARCHIVE+EXTRACTED+DERIVED+RESERVE) if stage=='archive' else DERIVED+RESERVE
    receipt=dict(start=start,end=time.monotonic(),stage=stage,device=os.stat(path).st_dev,free_bytes=usage.free,required_bytes=required,
        archive_bytes=ARCHIVE,extraction_bytes=EXTRACTED,derived_bytes=DERIVED,reserve_bytes=RESERVE,
        eligible=usage.free>=required,limitation='point-in-time free space, not a reservation against other writers')
    return receipt
