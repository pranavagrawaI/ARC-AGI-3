"""Prove Tinfield retains every BF16 PLE tensor from Qwen without fetching tables."""
import concurrent.futures as cf
import hashlib
import json
import hf_audit as h
import xet_diff as x

def main():
    p=h.ROOT/'azfocus/evidence'
    h.REPOS['source']='badtheorylabs/Tinfield-1'
    rev={'base':json.loads((p/'base/repo.json').read_text())['sha'],
         'source':json.loads((h.ROOT/'tinfield/evidence/source-repo.json').read_text())['sha']}
    identity=json.loads((h.ROOT/'tinfield/evidence/ple-identity.json').read_text())
    identical=json.loads((h.ROOT/'tinfield/evidence/source-shard-vs-base.json').read_text())['identical']
    needed=sorted(set(identity['ple'].values())-set(identical))
    jobs=[(k,rev[k],s) for k in rev for s in needed]
    with cf.ThreadPoolExecutor(4) as ex:list(ex.map(lambda j:x.reconstruction(*j),jobs))
    rows=[]
    for s in needed:
        a,b=[h.header(k,rev[k],s) for k in rev]
        sa=x.reconstruction('base',rev['base'],s);sb=x.reconstruction('source',rev['source'],s)
        # Permit a header-size shift while requiring identical term content.
        pos=0;lookup={}
        for t in sb['terms']:
            sig=(t['hash'],t['range']['start'],t['range']['end'],t['unpacked_length'])
            lookup.setdefault(sig,[]).append(pos);pos+=t['unpacked_length']
        pos=0;spans=[];shift=b['header_bytes']-a['header_bytes']
        for t in sa['terms']:
            sig=(t['hash'],t['range']['start'],t['range']['end'],t['unpacked_length'])
            if pos+shift in lookup.get(sig,[]):spans.append((pos,pos+t['unpacked_length']))
            pos+=t['unpacked_length']
        for n in [n for n,v in identity['ple'].items() if v==s]:
            assert a['tensors'][n]==b['tensors'][n]
            start,end=a['tensors'][n]['data_offsets'];offset=8+a['header_bytes']
            gaps=x.residual(offset+start,offset+end,spans);checks=[]
            for st,en in gaps:
                assert en-st<128*1024*1024,(n,en-st)
                da=h.ranged('base',rev['base'],s,st,en-1)
                db=h.ranged('source',rev['source'],s,st+shift,en-1+shift)
                assert da==db,('Tinfield changed PLE',n)
                checks.append({'span':[st,en],'sha256':hashlib.sha256(da).hexdigest()})
            rows.append({'name':n,'shard':s,'bytes':end-start,'xet_identical_spans':spans,'residual_checks':checks})
    identity.update(all_ple_tensors_identical=True,mixed_shard_checks=rows,proof='LFS SHA256 for pure PLE shards; Xet content addresses plus ranged residual equality for mixed shards')
    h.save(h.ROOT/'tinfield/evidence/ple-identity.json',identity)
    print('verified all',identity['ple_count'],'PLE tensors unchanged',flush=True)

if __name__=='__main__':main()
