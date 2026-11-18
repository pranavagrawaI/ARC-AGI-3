"""Pinned HF metadata and bounded ranged reads; never downloads a whole shard."""
import concurrent.futures as cf
import hashlib
import json
import struct
import time
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent
REPOS = {'base': 'Qwen/Qwen3.8-Flash-Next', 'az': 'AxonZeta/Qwen3.8-Flash-Next-AZ-Focus',
         'intel': 'Intel/Qwen3.8-Flash-Next-W4A16-AutoRound', 'tin': 'natnitaract/tinfield-1-nvfp4-probe'}

def save(p, d):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, indent=2) + '\n')

def get(url, **kw):
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=(15, 90), **kw)
            r.raise_for_status()
            return r
        except requests.RequestException:
            if attempt == 3: raise
            time.sleep(2)

def metadata(label):
    d = ROOT / ('tinfield' if label == 'tin' else 'azfocus') / 'evidence' / label
    info = get('https://huggingface.co/api/models/' + REPOS[label], params={'blobs': 'true'}).json()
    save(d / 'repo.json', info)
    rev = info['sha']
    for f in ['config.json', 'model.safetensors.index.json', 'README.md']:
        b = get(f'https://huggingface.co/{REPOS[label]}/resolve/{rev}/{f}').content
        (d / f).write_bytes(b)
    return info

def ranged(label, rev, shard, start, end):
    # Fail closed on a server ignoring Range; stream to avoid accidental full download.
    url = f'https://huggingface.co/{REPOS[label]}/resolve/{rev}/{shard}'
    with get(url, headers={'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'},
             params={'range_start': start, 'range_end': end}, stream=True) as r:
        assert r.status_code == 206, (r.status_code, url)
        assert r.headers.get('Content-Range', '').startswith(f'bytes {start}-{end}/'), r.headers
        out = bytearray()
        for chunk in r.iter_content(1024 * 1024):
            out.extend(chunk)
            assert len(out) <= end-start+1, 'oversized ranged response'
        assert len(out) == end-start+1
        return bytes(out)

def header(label, rev, shard):
    p = ROOT / ('tinfield' if label == 'tin' else 'azfocus') / 'evidence' / label / (shard+'.header.json')
    if p.exists(): return json.loads(p.read_text())
    n = struct.unpack('<Q', ranged(label, rev, shard, 0, 7))[0]
    assert 0 < n < 32*1024*1024
    h = json.loads(ranged(label, rev, shard, 8, 7+n))
    value = {'header_bytes': n, 'tensors': h}
    save(p, value)
    return value

def main():
    with cf.ThreadPoolExecutor(4) as ex:
        infos = dict(zip(REPOS, ex.map(metadata, REPOS)))
    maps = {k: {f['rfilename']:f for f in d['siblings']} for k,d in infos.items()}
    shards = [s for s in maps['base'] if s.endswith('.safetensors')]
    assert set(shards) == {s for s in maps['az'] if s.endswith('.safetensors')}
    changed = [s for s in shards if maps['base'][s]['lfs']['sha256'] != maps['az'][s]['lfs']['sha256']]
    save(ROOT/'azfocus/evidence/shard-diff.json', {'identical_shards': [s for s in shards if s not in changed], 'changed_shards': changed})
    print('changed shards', len(changed), changed, flush=True)
    jobs = [(k, infos[k]['sha'], s) for k in ['base','az'] for s in changed]
    jobs += [('intel', infos['intel']['sha'], s) for s in maps['intel'] if s.endswith('.safetensors')]
    with cf.ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda j: header(*j), jobs))
    candidates = []
    for s in changed:
        a = header('base', infos['base']['sha'], s)['tensors']
        b = header('az', infos['az']['sha'], s)['tensors']
        assert set(a)-{'__metadata__'} == set(b)-{'__metadata__'}
        for name in a:
            if name=='__metadata__': continue
            assert a[name] == b[name], (s,name)
            # Test every tensor in changed shards, rather than trusting the card's scope.
            candidates.append((s,name,a[name]))
    print('tensors requiring byte comparison',len(candidates),flush=True)
    def compare(job):
        s, name, tensor = job
        start,end=tensor['data_offsets']
        assert end-start <= 256*1024*1024, (name,end-start)
        hashes={}
        for k in ['base','az']:
            offset=8+header(k,infos[k]['sha'],s)['header_bytes']
            data=ranged(k,infos[k]['sha'],s,offset+start,offset+end-1)
            hashes[k]=hashlib.sha256(data).hexdigest()
            if k=='az' and hashes[k]!=hashes['base']:
                p=ROOT/'azfocus/evidence/slices'/f'{hashlib.sha256(name.encode()).hexdigest()}.bin'
                p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
        row={'name':name,'shard':s,'tensor':tensor,'hashes':hashes,'changed':hashes['base']!=hashes['az']}
        save(ROOT/'azfocus/evidence/tensor-comparison'/f'{hashlib.sha256(name.encode()).hexdigest()}.json',row)
        return row
    with cf.ThreadPoolExecutor(4) as ex:
        rows=list(ex.map(compare,candidates))
    intel={}
    for s in maps['intel']:
        if s.endswith('.safetensors'):
            for name,t in header('intel',infos['intel']['sha'],s)['tensors'].items():
                if name!='__metadata__':intel[name]={'shard':s,**t}
    for row in rows:
        if row['changed']:
            row['intel']=intel.get(row['name'])
            stem=row['name'].removesuffix('.weight')
            row['intel_quantized_keys']={k:v for k,v in intel.items() if k.startswith(stem+'.') and k!=row['name']}
    save(ROOT/'azfocus/evidence/tensor-diff.json', {'revisions':{k:v['sha'] for k,v in infos.items()},'compared_tensors':len(rows),'rows':rows})
    print('changed tensors',sum(r['changed'] for r in rows),flush=True)

if __name__=='__main__':main()
