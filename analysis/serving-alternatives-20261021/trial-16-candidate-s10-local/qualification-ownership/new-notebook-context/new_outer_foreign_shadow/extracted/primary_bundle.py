"""Small separately retrievable primary evidence; never replaces full raw qualification."""
import base64,hashlib,json,time
from pathlib import Path
MAX_BYTES=16*1024**2
NAME='000-serving-candidate16-primary.json'
REQUIRED=('result.json','supervisor.jsonl','observer-scope.json','host-boundaries.jsonl',
          'host-series.result.json','candidate-scope.json','candidate-release.json','query-compatibility.json')
PACKAGE=('installed-source-identity.json','model-source-identity.json','candidate-launch.json','prepared.json')

def record(data):
    return dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),base64=base64.b64encode(data).decode())


def write(root,config,end):
    started=time.monotonic();root=Path(root);files={};gaps=[];truncated=[]
    def add(path,key,limit,required=True):
        if time.monotonic()>end:raise TimeoutError('primary bundle deadline')
        if not path.exists():
            if required:gaps.append(key)
            return
        if path.stat().st_size>limit:
            if required:raise ValueError('required primary file bound: '+key)
            truncated.append(dict(path=key,bytes=path.stat().st_size,scope='last bytes only'))
            with path.open('rb') as stream:stream.seek(-limit,2);data=stream.read(limit)
        else:data=path.read_bytes()
        files[key]=record(data)
    for key in REQUIRED:add(root/key,key,4*1024**2)
    for key in PACKAGE:add(root/'candidate-package'/key,'candidate-package/'+key,512*1024)
    for key in ('candidate.server.stdout','candidate.server.stderr'):
        add(root/key,key,2*1024**2,required=False)
    add(root.parent/'config.json','config.json',128*1024)
    add(root.parent/'assignment.json','assignment.json',128*1024)
    health=[]
    for path in sorted(root.glob('candidate.health-*.*')):
        if path.suffix in ('.stdout','.stderr'):
            data=path.read_bytes();health.append(dict(path=path.name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    series=root/'host-series.jsonl';oom=[]
    if series.exists():
        digest=hashlib.sha256();total=0
        with series.open('rb') as stream:
            for line in stream:
                if time.monotonic()>end:raise TimeoutError('primary series deadline')
                digest.update(line);total+=len(line);row=json.loads(line)
                oom.append(dict(start=row['start'],end=row['end'],events_raw=row['raw']['events'],oom_events=row['oom_events']))
        series_ref=dict(bytes=total,sha256=digest.hexdigest(),samples=len(oom),projection='raw OOM counters only; full host/process series required in archive')
    else:gaps.append('host-series.jsonl');series_ref=None
    source={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in Path(__file__).parent.glob('*.py')}
    value=dict(schema='CANDIDATE16_PRIMARY_V1',run_id=config['run_id'],files=files,
        source_sha256=source,health_inventory=health,series_ref=series_ref,oom_projection=oom,
        missing=gaps,truncated=truncated,full_raw_required=True,capacity_or_rate_qualification=False)
    raw=json.dumps(value,sort_keys=True).encode()
    if len(raw)>MAX_BYTES:raise ValueError('primary bundle16MiB bound')
    if time.monotonic()>end:raise TimeoutError('primary bundle late before persistence')
    target=root.parent.parent/NAME
    with target.open('xb') as stream:stream.write(raw)
    receipt=dict(start=started,path=str(target),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),end=time.monotonic(),deadline=end)
    if receipt['end']>end:raise TimeoutError('primary bundle late persistence')
    (root/'primary-bundle-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


def verify(raw):
    if len(raw)>MAX_BYTES:raise ValueError('primary input bound')
    value=json.loads(raw)
    if value['schema']!='CANDIDATE16_PRIMARY_V1' or value['capacity_or_rate_qualification'] is not False:raise ValueError('primary schema')
    for key,item in value['files'].items():
        data=base64.b64decode(item['base64'],validate=True)
        if len(data)!=item['bytes'] or hashlib.sha256(data).hexdigest()!=item['sha256']:raise ValueError('primary bytes/hash '+key)
    return value
