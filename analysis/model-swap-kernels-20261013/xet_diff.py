"""Prove identical byte spans from Xet content addresses; range-read residual spans.

Protocol: https://huggingface.co/docs/xet/file-reconstruction
Saved evidence contains content addresses, never access tokens or signed fetch URLs.
"""
import concurrent.futures as cf
import hashlib
import json
import re
from pathlib import Path
import hf_audit as h

def reconstruction(label, rev, shard):
    p=h.ROOT/'azfocus/evidence'/label/(shard+'.xet.json')
    if p.exists(): return json.loads(p.read_text())
    r=h.get(f'https://huggingface.co/{h.REPOS[label]}/resolve/{rev}/{shard}',allow_redirects=False)
    links=r.headers['Link']
    auth=re.search('<([^>]+)>; rel="xet-auth"',links)[1]
    url=re.search('<([^>]+)>; rel="xet-reconstruction-info"',links)[1]
    token=h.get(auth).json()['accessToken']
    size=int(r.headers['X-Linked-Size'])
    d=h.get(url,headers={'Authorization':'Bearer '+token,'Range':f'bytes=0-{size-1}'}).json()
    assert d['offset_into_first_range']==0
    terms=d['terms']
    assert sum(t['unpacked_length'] for t in terms)==size
    out={'file_id':r.headers['X-Xet-Hash'],'lfs_sha256':r.headers['X-Linked-ETag'].strip('"'),'size':size,'terms':terms}
    h.save(p,out)
    return out

def shared_spans(a,b):
    # Same content-addressed xorb/chunk range, same byte length and file offset
    # identifies exactly the same bytes. Unmatched boundaries are read directly.
    def spans(d):
        pos=0;out={}
        for t in d['terms']:
            end=pos+t['unpacked_length']
            out[(pos,end)]=(t['hash'],t['range']['start'],t['range']['end'])
            pos=end
        return out
    x,y=spans(a),spans(b)
    return sorted((s,e) for (s,e),v in x.items() if y.get((s,e))==v)

def residual(start,end,spans):
    out=[];pos=start
    for s,e in spans:
        if e<=pos:continue
        if s>=end:break
        if s>pos:out.append((pos,min(s,end)))
        pos=max(pos,min(e,end))
    if pos<end:out.append((pos,end))
    return out

def main():
    p=h.ROOT/'azfocus/evidence'
    infos={k:json.loads((p/k/'repo.json').read_text()) for k in ['base','az','intel']}
    changed=json.loads((p/'shard-diff.json').read_text())['changed_shards']
    jobs=[(k,infos[k]['sha'],s) for k in ['base','az'] for s in changed]
    with cf.ThreadPoolExecutor(8) as ex:list(ex.map(lambda j:reconstruction(*j),jobs))
    proofs={s:shared_spans(reconstruction('base',infos['base']['sha'],s),reconstruction('az',infos['az']['sha'],s)) for s in changed}
    h.save(p/'xet-shared-spans.json',proofs)
    rows=[]
    for s in changed:
        header=h.header('base',infos['base']['sha'],s);offset=8+header['header_bytes']
        for n,t in header['tensors'].items():
            if n=='__metadata__':continue
            start,end=t['data_offsets']
            pieces=residual(offset+start,offset+end,proofs[s])
            rows.append({'name':n,'shard':s,'tensor':t,'residual_spans':pieces,'covered_bytes':end-start-sum(e-a for a,e in pieces)})
    print('residual bytes per repo',sum(sum(e-a for a,e in r['residual_spans']) for r in rows),flush=True)
    h.save(p/'xet-residual-plan.json',rows)
    def compare(row):
        n,s,t=row['name'],row['shard'],row['tensor']
        cp=p/'tensor-comparison'/f'{hashlib.sha256(n.encode()).hexdigest()}.json'
        if cp.exists(): return json.loads(cp.read_text())
        hashes={};different=False
        for start,end in row['residual_spans']:
            assert end-start<=256*1024*1024,(n,end-start)
            data={k:h.ranged(k,infos[k]['sha'],s,start,end-1) for k in ['base','az']}
            hashes[f'{start}:{end}']={k:hashlib.sha256(v).hexdigest() for k,v in data.items()}
            different |= data['base']!=data['az']
        row.update(changed=different,residual_sha256=hashes,identity_method='Xet shared spans plus SHA256 of all residual bytes')
        if different:
            assert t['dtype']=='BF16' and t['data_offsets'][1]-t['data_offsets'][0]<256*1024*1024
            offset=8+h.header('az',infos['az']['sha'],s)['header_bytes']
            data=h.ranged('az',infos['az']['sha'],s,offset+t['data_offsets'][0],offset+t['data_offsets'][1]-1)
            dest=p/'slices'/f'{hashlib.sha256(n.encode()).hexdigest()}.bin'
            dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
            row['az_tensor_sha256']=hashlib.sha256(data).hexdigest()
        h.save(cp,row);return row
    with cf.ThreadPoolExecutor(4) as ex:rows=list(ex.map(compare,rows))
    intel={}
    for f in (p/'intel').glob('*.header.json'):
        for n,t in json.loads(f.read_text())['tensors'].items():
            if n!='__metadata__':intel[n]={'shard':f.name.removesuffix('.header.json'),**t}
    for r in rows:
        if r['changed']:
            r['intel']=intel.get(r['name'])
            stem=r['name'].removesuffix('.weight')
            r['intel_quantized_keys']={k:v for k,v in intel.items() if k.startswith(stem+'.') and k!=r['name']}
    h.save(p/'tensor-diff.json',{'revisions':{k:v['sha'] for k,v in infos.items()},'compared_tensors':len(rows),'rows':rows})
    print('changed tensors',sum(r['changed'] for r in rows),flush=True)

if __name__=='__main__':main()
