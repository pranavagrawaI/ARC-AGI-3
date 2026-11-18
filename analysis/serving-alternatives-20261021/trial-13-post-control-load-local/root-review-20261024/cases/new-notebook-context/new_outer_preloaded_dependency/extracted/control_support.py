"""Exact shipping shared helpers extracted without candidate setup/imports."""
import csv,io,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path

def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def save(name,value):
    out=Path(os.environ['SCREEN_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
    (out/name).write_text(json.dumps(value,indent=2)+'\n')

def dataset(owner,slug):
    paths=[Path('/kaggle/input/datasets')/owner/slug,Path('/kaggle/input')/slug]
    found=[p for p in paths if p.is_dir()]
    if len(found)!=1:raise ValueError('missing/ambiguous exact mounted dataset '+owner+'/'+slug)
    return found[0]

def host_guard():
    inventory=subprocess.check_output(['nvidia-smi'],text=True,timeout=10)
    save('actual-nvidia-smi.json',dict(raw=inventory))
    query=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version,uuid',
                                   '--format=csv,noheader,nounits'],text=True,timeout=10)
    save('actual-gpu-query.json',dict(raw=query))
    rows=[row for row in csv.reader(io.StringIO(query)) if any(cell.strip() for cell in row)]
    if len(rows)!=1 or len(rows[0])!=4:raise ValueError('actual GPU query requires exactly one GPU/four fields')
    name,memory,driver,uuid=(value.strip() for value in rows[0])
    if name!='NVIDIA RTX PRO 6000 Blackwell Server Edition':raise ValueError('actual GPU identity mismatch')
    if not memory.isdigit() or int(memory)<90000:raise ValueError('actual GPU memory below90000MiB')
    if not driver or not uuid.startswith('GPU-'):raise ValueError('actual GPU driver/UUID identity missing')
    save('actual-gpu-query.json',dict(raw=query,name=name,memory_mib=int(memory),driver=driver,uuid=uuid))
    capability=inventory.split('CUDA Version:',1)[1].split()[0]
    if tuple(map(int,capability.split('.'))) < (13,0):raise ValueError('CUDA13 driver capability missing')
    meminfo=Path('/proc/meminfo').read_text();total=int(meminfo.split('MemTotal:',1)[1].split()[0])*1024
    raw=Path('/sys/fs/cgroup/memory.max').read_text().strip()
    limit=min(total,int(raw)) if raw.isdigit() else total
    free=shutil.disk_usage('/tmp').free
    save('host-headroom.json',dict(meminfo=meminfo,cgroup_memory_max=raw,effective_memory_bytes=limit,tmp_free_bytes=free))
    if limit<170*1024**3:raise ValueError('host memory below170GiB')
    if free<80*1024**3:raise ValueError('writable disk below80GiB')
