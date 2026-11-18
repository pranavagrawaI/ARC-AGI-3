"""Local simultaneous storage admission; no network/GPU/credentials."""
import os,shutil,time
from pathlib import Path
ARCHIVE=8*1024**3
EXTRACTED=8*1024**3
DERIVED=2*1024**3
RESERVE=1024**3

def check(path,stage='required-files'):
    path=Path(path);start=time.monotonic();usage=shutil.disk_usage(path)
    if stage not in ('required-files','assembly','derived'):raise ValueError('unknown local space stage')
    required={'required-files':ARCHIVE+EXTRACTED+DERIVED+RESERVE,'assembly':EXTRACTED+DERIVED+RESERVE,'derived':DERIVED+RESERVE}[stage]
    receipt=dict(start=start,end=time.monotonic(),stage=stage,device=os.stat(path).st_dev,free_bytes=usage.free,required_bytes=required,
        fetched_bytes=ARCHIVE,extraction_bytes=EXTRACTED,derived_bytes=DERIVED,reserve_bytes=RESERVE,
        eligible=usage.free>=required,limitation='point-in-time free space, not a reservation against other writers')
    return receipt
