import csv, hashlib, json, pathlib, subprocess, sys, time
CONFIG = {'id': 'pranavagrawai/serving-rtx-envelope-probe-20261021', 'title': 'serving-rtx-envelope-probe-20261021', 'code_file': 'serving-rtx-envelope-probe-20261021.ipynb', 'language': 'python', 'kernel_type': 'notebook', 'is_private': True, 'enable_gpu': True, 'enable_tpu': False, 'enable_internet': False, 'keywords': ['gpu'], 'dataset_sources': [], 'kernel_sources': [], 'competition_sources': ['arc-prize-2026-arc-agi-3'], 'model_sources': [], 'docker_image': 'gcr.io/kaggle-private-byod/python@sha256:57e612b484cf3df5026ee4dcc3cb176974b22b2bc0937fb1e16132a8be4cb13c', 'machine_shape': 'NvidiaRtxPro6000'}
OUT=pathlib.Path('/kaggle/working/probe.json')
def classify(query):
    if query['returncode'] != 0: return False
    try:
        rows=list(csv.reader(query['stdout'].strip().splitlines()))
        return len(rows)==1 and 'rtx pro 6000' in rows[0][0].lower() and int(rows[0][1].strip())>=90000
    except (ValueError,IndexError): return False
def command(args):
    try:
        r=subprocess.run(args,capture_output=True,text=True,timeout=15)
        return {'command':args,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
    except Exception as e:
        return {'command':args,'returncode':None,'stdout':'','stderr':repr(e)}
record={'probe_id':CONFIG['id'],'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'requested_metadata':CONFIG,'requested_metadata_sha256':hashlib.sha256(json.dumps(CONFIG,sort_keys=True).encode()).hexdigest(),
        'python':sys.version,'no_runtime_install':True,'no_model_load':True}
record['nvidia_smi']=command(['nvidia-smi'])
record['gpu_query']=command(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader,nounits'])
record['pass']=bool(record['nvidia_smi']['returncode']==0 and classify(record['gpu_query']))
for key,path in [('cgroup_memory_max','/sys/fs/cgroup/memory.max'),('os_release','/etc/os-release')]:
    p=pathlib.Path(path)
    record[key]=p.read_text() if p.exists() else None
OUT.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2),flush=True)
assert record['pass'], 'allocation probe failed; stop GPU work; no retry'
