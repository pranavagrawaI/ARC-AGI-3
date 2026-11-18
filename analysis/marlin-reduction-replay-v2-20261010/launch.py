"""One authorized push. Admission errors never consume a GPU attempt."""
import datetime as dt
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
SLUG='pranavagrawai/'+HERE.name
HARD_TIMEOUT_S=3600
WEEKLY_CAP_S=108000
INVALID=['invalidTags','invalidDatasetSources','invalidCompetitionSources','invalidKernelSources','invalidModelSources']


def now():return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def verify_local(root=HERE):
    root=Path(root)
    frozen=json.loads((root/'launch-freeze.json').read_text())
    prepared=json.loads((root/'prepared-manifest.json').read_text())
    for name,digest in {**prepared['files'],**frozen['files']}.items():
        if sha(root/name)!=digest:raise ValueError('Frozen artifact changed: '+name)
    sdk=json.loads((root/'SDK-CONTRACT.json').read_text())
    for name,digest in sdk['files'].items():
        if sha(Path(name))!=digest:raise ValueError('Installed SDK source changed before launch')
    authority=json.loads((root/'AUTHORITY.json').read_text())
    if authority['instruction']!='do the launch setup, then then launch it.' or authority['scope']!='one Marlin paired replay qualification' or authority['hard_timeout_s']!=3600 or authority['gpu_count']!=1 or authority['run_id']!=root.name:
        raise ValueError('Authority differs from the single bounded replay')
    metadata=json.loads((root/'kernel/kernel-metadata.json').read_text())
    baseline=json.loads((root.parent/'serving-headroom-v3-20261008/kernel/kernel-metadata.json').read_text())
    for field in ['dataset_sources','model_sources','competition_sources','docker_image','machine_shape','is_private','enable_gpu','enable_tpu','enable_internet','kernel_sources','language','kernel_type']:
        if metadata[field]!=baseline[field]:raise ValueError('Qualified attachment/hardware metadata changed: '+field)
    if metadata['id']!=SLUG or metadata['code_file']!=root.name+'.ipynb':raise ValueError('Wrong run identity')
    # Entire wheel/lock and inherited payload readback remain bound to qualification.
    base=root.parent/'serving-headroom-v3-20261008'
    contract=json.loads((base/'native-contract.json').read_text())
    wheel=base/'evidence/packages'/contract['wheel']['filename']
    if sha(wheel)!=contract['wheel']['sha256'] or sha(base/'evidence/packages/requirements.lock')!=contract['wheel']['requirements_lock_sha256']:
        raise ValueError('Qualified wheel or dependency lock changed')
    data=root.parent/'mtp-depth-20261006/kaggle/dataset'
    manifest=json.loads((data/'manifest.json').read_text())
    readback=json.loads((root.parent/'mtp-depth-20261006/evidence/dataset-readback.json').read_text())
    if readback.get('mismatches') or readback['files']!=manifest or readback['manifest_sha256']!=sha(data/'manifest.json'):
        raise ValueError('Inherited uploaded payload readback changed')
    for name,digest in manifest.items():
        if sha(data/name)!=digest:raise ValueError('Inherited replay payload changed: '+name)
    return {'slug':SLUG,'base_commit':'ce99ac6caef4a8fa2eb39f37cf0e7233ee157dda','launch_freeze_sha256':sha(root/'launch-freeze.json'),
            'card_sha256':sha(root/'EXPERIMENT-CARD.md'),'notebook_sha256':sha(root/'kernel'/metadata['code_file']),'hard_timeout_s':3600,'authority':authority}


def quota_receipt(response):
    quota=getattr(response,'gpu_quota',None)
    used=getattr(quota,'time_used',None);reserved=getattr(quota,'time_reserved',None)
    if not isinstance(used,dt.timedelta) or not isinstance(reserved,dt.timedelta):raise ValueError('Typed quota durations missing')
    values=[used.total_seconds(),reserved.total_seconds()]
    if min(values)<0:raise ValueError('Negative quota duration')
    receipt={'utc':now(),'used_seconds':values[0],'reserved_seconds':values[1],'new_cap_seconds':3600,'weekly_cap_seconds':108000}
    if sum(values)+HARD_TIMEOUT_S>WEEKLY_CAP_S:raise ValueError('Quota does not fit the full one-hour cap')
    return receipt


def preflight(backend,root=HERE):
    identity=verify_local(root)
    quota=quota_receipt(backend.quota())
    active=backend.active_sessions(SLUG.split('/')[0])
    if active:raise ValueError('Competing or unknown kernel state; cannot admit isolated replay')
    backend.require_new_slug(SLUG)
    receipt={'utc':now(),'identity':identity,'quota':quota,'active_sessions':active,'admitted':True}
    save(Path(root)/'evidence/quota-before.json',receipt)
    return receipt


def accepted_response(response):
    return (response.get('error') is None and type(response.get('versionNumber')) is int and response['versionNumber']==1
            and type(response.get('kernelId')) is int and response['kernelId']>0
            and response.get('url')=='https://www.kaggle.com/code/'+SLUG
            and all(response.get(key)==[] for key in INVALID))


def push_once(backend,admission,root=HERE):
    root=Path(root);attempt=root/'evidence/launch-attempt.json'
    # Recheck local identity immediately at the mutation boundary.
    identity=verify_local(root)
    if identity!=admission['identity']:raise ValueError('Identity changed since live admission')
    row={'utc':now(),'state':'attempting','identity':identity,'push_calls':1}
    with attempt.open('x') as stream:stream.write(json.dumps(row,indent=2)+'\n')
    try:
        response=backend.push(root/'kernel',HARD_TIMEOUT_S)
        save(root/'evidence/launch-response.json',response)
        if not accepted_response(response):raise ValueError('Push response not a complete first-version acceptance')
        row.update(state='accepted',response=response)
        return response
    except Exception as error:
        row.update(state='acceptance unresolved; no retry',error_type=type(error).__name__)
        raise
    finally:save(attempt,row)
