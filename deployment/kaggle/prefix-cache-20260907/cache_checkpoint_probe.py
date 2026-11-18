"""CPU execution of pinned checkpoint-position arithmetic, not a GPU cache test."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
import zipfile

METHOD='_mamba_radix_cache_v2_req_prepare_for_extend'
SOURCE='sglang/srt/managers/schedule_batch.py'


def inspect(wheel, raw):
    with zipfile.ZipFile(wheel) as archive:
        source=archive.read(SOURCE)
    tree=ast.parse(source)
    method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==METHOD)
    # Keep function body verbatim; deferred annotations avoid loading serving dependencies.
    definition=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),method],type_ignores=[])
    env={'mamba_cache_chunk_size':lambda:64,'mamba_checkpoint_grid':lambda page:64,
         'mamba_extra_buffer_lazy_enabled':lambda:False,'_MambaRadixCacheV2TrackEntry':lambda **kw:NS(**kw)}
    exec(compile(ast.fix_missing_locations(definition),SOURCE,'exec'),env)
    fn=env[METHOD]
    owner=NS(model_config=NS(hf_text_config=NS()),tree_cache=NS(page_size=1),
             req_to_token_pool=NS(get_mamba_ping_pong_other_idx=lambda index:1-index))
    def req(prefix,length,branch=None):
        return NS(prefix_indices=[0]*prefix,extend_range=NS(length=length),
                  mamba_ping_pong_track_buffer=[NS(item=lambda:0),NS(item=lambda:1)],
                  mamba_next_track_idx=0,mamba_branching_seqlen=branch)
    results=[]
    for i,chunk in enumerate((2048,4096,4096,2048)):
        arm=raw/f'arm-{i}-chunk-{chunk}'
        info=json.loads((arm/'server-info.json').read_text())
        assert info['page_size']==1 and info['mamba_radix_cache_strategy']=='extra_buffer'
        for pair in json.loads((arm/'inputs.json').read_text()):
            checkpoints=[]
            for start in range(0,pair['seed_tokens'],chunk):
                r=req(start,min(chunk,pair['seed_tokens']-start))
                entry=fn(owner,r)
                if entry.track_mask:checkpoints.append(r.mamba_last_track_seqlen)
            predicted=max([n for n in checkpoints if n<=pair['lcp_tokens']],default=0)
            actual=json.loads((arm/(pair['name']+'-predecessor-condition.json')).read_text())['successor']['cached_input_tokens']
            assert predicted==actual,(i,pair['name'],predicted,actual)
            results.append({'arm':i,'chunk':chunk,'pair':pair['name'],'lcp':pair['lcp_tokens'],
                            'checkpoint_positions':checkpoints,'predicted_reusable':predicted,'observed_cached':actual})
    # A known branching point can create a checkpoint inside a large prefill chunk.
    r=req(16384,4096,19648)
    entry=fn(owner,r)
    assert r.mamba_last_track_seqlen==19648
    return {'wheel_sha256':hashlib.sha256(wheel.read_bytes()).hexdigest(),
            'source':SOURCE,'source_sha256':hashlib.sha256(source).hexdigest(),
            'method_lines':[method.lineno,method.end_lineno],
            'assumptions':['64-token checkpoint grid from pinned model-state fallback and page1',
                           'cold predecessor processed in configured prefill chunks; donated checkpoints survive',
                           'CPU fixtures do not execute state copies, kernels, eviction or scheduler overlap'],
            'matches':results,'branching_control':{'prefix':16384,'extend':4096,'branch':19648,
                'tracked_checkpoint':r.mamba_last_track_seqlen,'kernel_track_seqlen':entry.track_seqlen},
            'conclusion':'Observed chunk-aligned hits match checkpoint availability; chunk-size rounding is not a universal cache matching rule.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('wheel',type=Path);parser.add_argument('raw',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    result=inspect(args.wheel,args.raw)
    with args.out.open('x') as handle:json.dump(result,handle,indent=2);handle.write('\n')
    print(json.dumps({'matched_cases':len(result['matches']),'branching_control':result['branching_control']}))
