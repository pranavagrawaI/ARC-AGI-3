"""Assemble public Tinfield NVFP4 core with byte-identical incumbent BF16 PLE.

No quantization, training, or checkpoint mutation. Pure PLE shards are linked;
mixed incumbent shards are sliced into new PLE-only safetensors by streaming.
"""
import hashlib
import json
import shutil
import struct
from pathlib import Path

def read_header(path):
    with path.open('rb') as f:
        n=struct.unpack('<Q',f.read(8))[0]
        assert 0<n<64*1024*1024
        return n,json.loads(f.read(n))

def ple_inventory(base_map, core_map):
    is_ple=lambda n:'.ngram_embedding.shard_' in n and n.endswith('.weight')
    ple={k:v for k,v in base_map.items() if is_ple(k)}
    expected={k:v for k,v in core_map.items() if is_ple(k)}
    assert len(ple)==128 and (not expected or set(ple)==set(expected)),(len(ple),len(expected))
    return ple

def select_shard(source,dest,names):
    n,header=read_header(source)
    assert set(names)<=set(header)
    if set(names)==set(header)-{'__metadata__'}:
        dest.symlink_to(source.resolve());return 'link'
    subset={};pos=0
    for name in sorted(names):
        t=header[name];size=t['data_offsets'][1]-t['data_offsets'][0]
        subset[name]={**t,'data_offsets':[pos,pos+size]};pos+=size
    raw=json.dumps(subset,separators=(',',':')).encode();raw+=b' '*((-len(raw))%8)
    with dest.open('xb') as out,source.open('rb') as src:
        out.write(struct.pack('<Q',len(raw)));out.write(raw)
        for name in sorted(names):
            st,en=header[name]['data_offsets'];src.seek(8+n+st);remaining=en-st
            while remaining:
                block=src.read(min(8*1024*1024,remaining));assert block
                out.write(block);remaining-=len(block)
    return 'streamed extraction'

def assemble(core,base,view):
    core,base,view=map(Path,(core,base,view));view.mkdir(parents=True,exist_ok=False)
    ci=json.loads((core/'model.safetensors.index.json').read_text())
    bi=json.loads((base/'model.safetensors.index.json').read_text())
    is_ple=lambda n:'.ngram_embedding.shard_' in n and n.endswith('.weight')
    # Public Kaggle core is intentionally incomplete: its index omits PLE.
    # The full HF export indexes PLE, so accept either documented core layout.
    ple=ple_inventory(bi['weight_map'],ci['weight_map'])
    cfg=json.loads((core/'config.json').read_text());bcfg=json.loads((base/'config.json').read_text())
    assert cfg['quantization_config']['quant_algo']=='NVFP4'
    for field in ['hidden_size','num_experts','num_experts_per_tok','num_hidden_layers','vocab_size']:
        assert cfg['text_config'][field]==bcfg['text_config'][field],field
    assert cfg['text_config']['num_experts']==512
    for name in ['tokenizer.json']:
        a,b=[json.loads((p/name).read_text()) for p in (core,base)]
        assert a['model']['vocab']==b['model']['vocab'] and a['added_tokens']==b['added_tokens'],name
        normalize=lambda merges:[m.split(' ') if isinstance(m,str) else m for m in merges]
        assert normalize(a['model']['merges'])==normalize(b['model']['merges']),name
    wm={};receipt={'core':str(core),'ple':str(base),'format':'ModelOpt NVFP4 experts + original BF16 PLE','ple_shards':[]}
    for shard in sorted({v for k,v in ci['weight_map'].items() if not is_ple(k)}):
        source=core/shard;assert source.is_file(),source
        _,header=read_header(source)
        assert not any(is_ple(k) for k in header),('core shard mixes PLE',shard)
        (view/shard).symlink_to(source.resolve())
        wm.update({k:shard for k,v in ci['weight_map'].items() if v==shard and not is_ple(k)})
    for i,shard in enumerate(sorted(set(ple.values()))):
        names=[k for k,v in ple.items() if v==shard];source=base/shard
        _,header=read_header(source)
        for name in names:
            assert header[name]['dtype']=='BF16' and header[name]['shape']==[2500012,160],(name,header[name])
        filename=f'incumbent-ple-{i:03d}.safetensors'
        how=select_shard(source,view/filename,names)
        wm.update({name:filename for name in names})
        receipt['ple_shards'].append({'source':shard,'tensors':len(names),'method':how})
        print('TINFIELD_PLE',shard,how,flush=True)
    assert set(wm)==set(ci['weight_map'])|set(ple),(len(wm),len(ci['weight_map']))
    # Keep every non-weight behavior asset from the named incumbent.
    for source in base.iterdir():
        if source.is_file() and source.suffix!='.safetensors' and source.name not in ['config.json','model.safetensors.index.json','hf_quant_config.json']:
            (view/source.name).symlink_to(source.resolve())
    (view/'hf_quant_config.json').symlink_to((core/'hf_quant_config.json').resolve())
    bcfg['quantization_config']=cfg['quantization_config']
    cfg=bcfg
    (view/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    (view/'model.safetensors.index.json').write_text(json.dumps({'metadata':ci.get('metadata',{}),'weight_map':wm})+'\n')
    receipt.update(tensor_count=len(wm),ple_tensor_count=len(ple),target_experts=512,draft_experts=512,
                   config_sha256=hashlib.sha256((view/'config.json').read_bytes()).hexdigest())
    Path('/kaggle/working/model-swap-assembly.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('TINFIELD_ASSEMBLED',json.dumps(receipt),flush=True)
    return str(view)
