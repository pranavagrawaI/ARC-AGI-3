"""Trusted corpus construction, independent of producer assertions."""
import base64
import hashlib
import importlib.util
import json
import struct
from pathlib import Path

MANIFEST_SHA = 'c21852dc33715d0efe6b4d62b70d7869040b859b4f1c2ec36ad1ea22e64b9713'
HERE = Path(__file__).resolve().parent
DATA = HERE.parents[2] / 'analysis/mtp-depth-20261006/kaggle/dataset'
RENDER_AUDIT_SHA='903e88e87f018bb2b113c59fd9d99e228f7b83af07e532fa1165500ffac06fe3'
CATALOG_SHA='dfc1654de8a324f8142306e6459fcc3c47cee6ef2768f91f8a10dd3d0aeb2985'


def verify_corpus(directory=DATA):
    directory=Path(directory)
    raw=(directory/'manifest.json').read_bytes()
    if digest(raw)!=MANIFEST_SHA:
        raise ValueError('immutable corpus manifest mismatch')
    for name, expected in json.loads(raw).items():
        with (directory/name).open('rb') as stream:
            actual=hashlib.file_digest(stream,'sha256').hexdigest()
        if actual!=expected:
            raise ValueError('immutable corpus file mismatch: '+name)
    return MANIFEST_SHA


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def image_observations(body):
    result=[]
    for message in body['messages']:
        content=message.get('content')
        if not isinstance(content,list):continue
        for part in content:
            if part.get('type')!='image_url':continue
            value=part['image_url'];url=value['url'] if isinstance(value,dict) else value
            header,encoded=url.split(',',1)
            if header!='data:image/png;base64':raise ValueError('unsupported source image')
            blob=base64.b64decode(encoded,validate=True)
            if blob[:8]!=b'\x89PNG\r\n\x1a\n':raise ValueError('invalid PNG source')
            width,height=struct.unpack('>II',blob[16:24])
            if (width,height)!=(640,640):raise ValueError('unexpected image geometry')
            result.append(digest(blob))
    return result


def wire_hash(body):
    value=dict(body);value.pop('model',None)
    return digest(canonical(value))


def payload_for(record,directory=DATA):
    policy=module('capacity_payload',Path(directory)/'payload.py')
    with (Path(directory)/record['file']).open('rb') as stream:
        stream.seek(record['offset']);row=json.loads(stream.readline())
    return policy.build(row,'flashnext',record['cap'])


def build(directory=DATA):
    directory=Path(directory);verify_corpus(directory)
    simulator=module('capacity_sim',directory/'sim.py')
    policy=module('capacity_payload',directory/'payload.py')
    games=simulator.build(json.loads((directory/'workload.json').read_text()))
    holders,queue=simulator.initial_admission(games,10)
    audit_path=HERE.parent/'trial-3-matched-preparation/processor-fit-records.jsonl'
    audit_raw=audit_path.read_bytes()
    if digest(audit_raw)!=RENDER_AUDIT_SHA:raise ValueError('local render audit identity mismatch')
    audit={row['game']+':'+str(row['index']):row for row in map(json.loads,audit_raw.splitlines())}
    records={};streams={}
    for game,value in sorted(games.items()):
        selected={x['line_index']:x for x in value['reqs']}
        path=next(directory.glob(game+'*_p0_requests.jsonl'))
        index=0;streams[game]=[]
        with path.open('rb') as stream:
            while True:
                offset=stream.tell();line=stream.readline()
                if not line:break
                row=json.loads(line)
                if row.get('event')!='request':continue
                source=selected.get(index);index+=1
                if source is None:continue
                body=policy.build(row,'flashnext',source['completion_tokens'])
                key=game+':'+str(source['line_index'])
                images=image_observations(body)
                proof=audit[key]
                if (proof['recorded_prompt_tokens']!=source['prompt_tokens'] or proof['images']!=len(images)
                        or proof['completion_cap']!=source['completion_tokens']
                        or proof['render_sha256']['control']!=proof['render_sha256']['candidate']):
                    raise ValueError('render audit/source join mismatch')
                records[key]=dict(game=game,index=source['line_index'],file=path.name,offset=offset,
                                  prompt=source['prompt_tokens'],cap=source['completion_tokens'],
                                  cached=source['cached_tokens'],images=images,
                                  wire_sha=wire_hash(body),extends_previous=source['extends_previous'],
                                  render_sha=proof['render_sha256']['control'],local_prompt=proof['prompt_tokens']['control'])
                streams[game].append(key)
    return dict(schema='capacity-corpus-v1',manifest_sha=MANIFEST_SHA,render_audit_sha=RENDER_AUDIT_SHA,records=records,streams=streams,
                initial_holders=holders,initial_queue=queue,slots=10)


def load(path=HERE/'catalog.json',expected_sha=CATALOG_SHA):
    raw=Path(path).read_bytes()
    if expected_sha is not None and digest(raw)!=expected_sha:raise ValueError('catalog identity mismatch')
    value=json.loads(raw)
    if value['manifest_sha']!=MANIFEST_SHA or value['slots']!=10:raise ValueError('catalog binding mismatch')
    return value


if __name__=='__main__':
    value=build();out=HERE/'catalog.json';out.write_bytes(canonical(value))
    print(json.dumps(dict(records=len(value['records']),games=len(value['streams']),sha256=digest(out.read_bytes()))))
