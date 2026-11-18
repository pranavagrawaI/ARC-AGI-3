"""Small separately retrievable primary evidence; never replaces full raw qualification."""
import base64,hashlib,json,math,time,zlib
from pathlib import Path
MAX_BYTES=16*1024**2
NAME='000-serving-pair17-primary.json'
MAX_DECODED=128*1024**2
SCHEMA='CANDIDATE_FIRST_PAIR17_PRIMARY_V1'
REQUIRED=('result.json','supervisor.jsonl','observer-scope.json','host-boundaries.jsonl',
          'host-series.result.json','candidate-scope.json','candidate-release.json','control-admission.json',
          'query-compatibility.json','control-server-binding.json','candidate-server-binding.json',
          'native-lease.json','release-scope.json','pre-release.json','release.json',
          'candidate/events.jsonl','control/events.jsonl','paired-result.json')
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
    for key in REQUIRED:add(root/key,key,32*1024**2)
    for key in PACKAGE:add(root/'candidate-package'/key,'candidate-package/'+key,512*1024)
    for key in ('control-prepared.json','control-source-identity.json','control-model-identity.json',
                'control-frontend-registration.json','control-health-identity.json','control-health-serve.json','control-package-identity.json'):
        add(root/'control-package'/key,'control-package/'+key,512*1024)
    for arm in ('candidate','control'):
        for key in ('actual-gpu-query.json','host-headroom.json'):
            add(root/(arm+'-package')/key,arm+'-package/'+key,128*1024)
    for key in ('candidate.server.stdout','candidate.server.stderr','control.server.stdout'):
        add(root/key,key,64*1024,required=False)
    add(root.parent/'config.json','config.json',128*1024)
    add(root.parent/'assignment.json','assignment.json',128*1024)
    health=[]
    for path in sorted(list(root.glob('candidate.health-*.*'))+list(root.glob('control.health-*.*'))):
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
    value=dict(schema=SCHEMA,run_id=config['run_id'],files=files,
        source_sha256=source,health_inventory=health,series_ref=series_ref,oom_projection=oom,
        missing=gaps,truncated=truncated,full_raw_required=True,capacity_or_rate_qualification=False)
    raw=encode(value)
    if len(raw)>MAX_BYTES:raise ValueError('primary bundle16MiB bound')
    if time.monotonic()>end:raise TimeoutError('primary bundle late before persistence')
    target=root.parent.parent/NAME
    with target.open('xb') as stream:stream.write(raw)
    receipt=dict(schema='PRIMARY_PUBLICATION_V2',start=started,path=str(target),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),end=time.monotonic(),deadline=end)
    if receipt['end']>end:raise TimeoutError('primary bundle late persistence')
    (root/'primary-bundle-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    receipt['persisted_end']=time.monotonic()
    if receipt['persisted_end']>end:raise TimeoutError('primary receipt persistence completed late')
    return receipt


def encode(value):
    decoded=json.dumps(value,sort_keys=True,separators=(',',':')).encode()
    if len(decoded)>MAX_DECODED:raise ValueError('primary decoded bound')
    data=zlib.compress(decoded,9)
    envelope=dict(schema='PAIR17_PRIMARY_ZLIB_V1',decoded_bytes=len(decoded),decoded_sha256=hashlib.sha256(decoded).hexdigest(),
        compressed_bytes=len(data),compressed_sha256=hashlib.sha256(data).hexdigest(),compressed_base64=base64.b64encode(data).decode())
    raw=json.dumps(envelope,sort_keys=True,separators=(',',':')).encode()
    if len(raw)>MAX_BYTES:raise ValueError('primary16MiB encoded bound')
    return raw


def verify(raw):
    if len(raw)>MAX_BYTES:raise ValueError('primary input bound')
    envelope=json.loads(raw)
    if envelope['schema']!='PAIR17_PRIMARY_ZLIB_V1' or type(envelope['decoded_bytes']) is not int or not 0<=envelope['decoded_bytes']<=MAX_DECODED:raise ValueError('primary envelope/bound')
    data=base64.b64decode(envelope['compressed_base64'],validate=True)
    if len(data)!=envelope['compressed_bytes'] or hashlib.sha256(data).hexdigest()!=envelope['compressed_sha256']:raise ValueError('primary compressed hash')
    decoder=zlib.decompressobj()
    try:decoded=decoder.decompress(data,MAX_DECODED+1)
    except zlib.error as error:raise ValueError('primary compression') from error
    if len(decoded)>MAX_DECODED or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:raise ValueError('primary compression bound/truncation/trailing data')
    if len(decoded)!=envelope['decoded_bytes'] or hashlib.sha256(decoded).hexdigest()!=envelope['decoded_sha256']:raise ValueError('primary decoded hash')
    value=json.loads(decoded)
    if value['schema']!=SCHEMA or value['capacity_or_rate_qualification'] is not False:raise ValueError('primary schema')
    for key,item in value['files'].items():
        if Path(key).is_absolute() or '..' in Path(key).parts:raise ValueError('primary unsafe file')
        content=base64.b64decode(item['base64'],validate=True)
        if len(content)!=item['bytes'] or hashlib.sha256(content).hexdigest()!=item['sha256']:raise ValueError('primary bytes/hash '+key)
    return value


def verify_publication(root,raw,work_end,seconds):
    """Independent durable joins; reject an otherwise timely but stale bundle."""
    root=Path(root);bundle=verify(raw)
    receipt_path=root/'primary-bundle-receipt.json'
    receipt_bytes=receipt_path.read_bytes();receipt=json.loads(receipt_bytes)
    completed=json.loads((root/'primary-publication-completion.json').read_bytes())
    clocks=[receipt['start'],receipt['end'],receipt['deadline'],completed['helper_persisted_end'],completed['returned_at'],work_end,seconds]
    if not all(type(value) in (int,float) and math.isfinite(value) for value in clocks):raise ValueError('publication finite clocks')
    if (receipt['schema']!='PRIMARY_PUBLICATION_V2' or completed['schema']!='PRIMARY_CALLER_COMPLETION_V2'
        or receipt['sha256']!=hashlib.sha256(raw).hexdigest() or receipt['bytes']!=len(raw)
        or completed['receipt_sha256']!=hashlib.sha256(receipt_bytes).hexdigest()
        or completed['deadline']!=receipt['deadline']
        or not receipt['start']<=receipt['end']<=completed['helper_persisted_end']<=completed['returned_at']<=receipt['deadline']<=min(work_end,receipt['start']+seconds)
        or Path(receipt['path']).name!=NAME):raise ValueError('primary publication completion/deadline join')
    actual=json.loads((root/'result.json').read_bytes())
    published=json.loads(base64.b64decode(bundle['files']['result.json']['base64'],validate=True))
    if actual!=published or actual.get('primary_bundle_error') or actual.get('late_work'):raise ValueError('stale primary result after publication failure')
    return dict(helper_persisted_end=completed['helper_persisted_end'],returned_at=completed['returned_at'],deadline=receipt['deadline'])
