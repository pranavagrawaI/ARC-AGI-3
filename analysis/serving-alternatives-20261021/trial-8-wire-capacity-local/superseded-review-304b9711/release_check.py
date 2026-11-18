"""Required empty GPU/port and reclaimable host headroom between packages."""
import csv
import io
import json
from pathlib import Path
import shutil
import socket
import subprocess

def validate(gpu,apps,meminfo,maximum,current,stat,disk):
    rows=list(csv.reader(io.StringIO(gpu)))
    if len(rows)!=1 or len(rows[0])!=3:raise ValueError('one complete GPU release query')
    name,memory,uuid=[x.strip() for x in rows[0]]
    if name!='NVIDIA RTX PRO 6000 Blackwell Server Edition' or not uuid.startswith('GPU-'):raise ValueError('release GPU identity')
    if apps.strip():raise ValueError('GPU compute process remains')
    if not memory.isdigit() or int(memory)>2048:raise ValueError('GPU memory not released below2048MiB')
    mem={k:int(v.split()[0])*1024 for k,v in (line.split(':',1) for line in meminfo.splitlines())}
    fields={k:int(v) for k,v in (line.split() for line in stat.splitlines())}
    available=mem['MemAvailable']
    if maximum.isdigit():available=min(available,max(0,int(maximum)-int(current)+fields.get('inactive_file',0)))
    if available<140*1024**3:raise ValueError('remaining reclaimable host headroom below140GiB')
    if disk<100*1024**3:raise ValueError('remaining writable disk below100GiB')
    return dict(name=name,uuid=uuid,gpu_memory_used_mib=int(memory),reclaimable_host_bytes=available,tmp_free_bytes=disk,
                inactive_file_accounting='cgroup max-current+inactive_file capped by host MemAvailable; eligibility only, not cache matching')

def run(output):
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,uuid','--format=csv,noheader,nounits'],text=True,timeout=10)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits'],text=True,timeout=10)
    raw=dict(gpu=gpu,apps=apps,meminfo=Path('/proc/meminfo').read_text(),maximum=Path('/sys/fs/cgroup/memory.max').read_text().strip(),
             current=Path('/sys/fs/cgroup/memory.current').read_text().strip(),stat=Path('/sys/fs/cgroup/memory.stat').read_text(),disk=shutil.disk_usage('/tmp').free)
    Path(output).write_text(json.dumps({'raw':raw},indent=2)+'\n')
    result=validate(**raw)
    free_port()
    Path(output).write_text(json.dumps({'raw':raw,'pass':True,'observed':result,'port8001_free':True},indent=2)+'\n')

def free_port(port=8001):
    with socket.socket() as check:
        check.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        check.bind(('127.0.0.1',port))

if __name__=='__main__':
    import sys
    run(sys.argv[1])
