"""GPU worker, only after separate launch authority. Capture files are local/trusted."""
import argparse
import contextlib
import types
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from reference import dequantize, errors, numerical_gate, paired_summary, qualify_outputs, unpack_marlin, unpermute_scales


def cpu_array(tensor):
    return tensor.detach().float().cpu().numpy()


def fingerprint(tensor):
    return hashlib.sha256(tensor.detach().contiguous().view(__import__('torch').uint8).cpu().numpy().tobytes()).hexdigest()


def verify_pack_sentinel():
    import torch
    from sglang.kernels.ops.quantization.gptq_marlin_repack import gptq_marlin_repack
    from sglang.srt.layers.quantization.marlin_utils import marlin_permute_scales
    # Known integers, not values recovered from GEMM outputs; every W4 code appears.
    rng = np.random.default_rng(20261009)
    q = rng.integers(0,16,size=(128,128),dtype=np.int32)
    packed = np.zeros((16,128),dtype=np.uint32)
    for i in range(8): packed |= q[i::8].astype(np.uint32) << (4*i)
    repacked = gptq_marlin_repack(torch.from_numpy(packed.view(np.int32)).cuda(), torch.empty(0,dtype=torch.int32,device='cuda'),128,128,4)
    np.testing.assert_array_equal(unpack_marlin(repacked.cpu().numpy(),128,128),q-8)
    for groups in [1,2]:
        original = torch.arange(groups*128,dtype=torch.float32,device='cuda').reshape(groups,128)
        result = marlin_permute_scales(original,128,128,128//groups if groups>1 else -1)
        np.testing.assert_array_equal(unpermute_scales(cpu_array(result),128),cpu_array(original))
    return {'integer_codes':16,'shape':[128,128],'scale_groups':[1,2],'passed':True}


def fp32_reference(sample, weights):
    import torch
    x = sample['hidden_states'].cuda().float()
    ids = sample['topk_ids'].cuda().long()
    route = sample['topk_weights'].cuda().float()
    output = torch.zeros_like(x)
    k = x.shape[1]; n = weights['w2'].shape[1]*16
    for expert in torch.unique(ids).tolist():
        if expert < 0 or expert >= weights['w1'].shape[0]:
            raise ValueError('Invalid local expert id')
        tokens, slots = torch.where(ids == expert)
        w1 = torch.from_numpy(dequantize(weights['w1'][expert].numpy(),cpu_array(weights['w1_scale'][expert]),k,2*n)).cuda()
        w2 = torch.from_numpy(dequantize(weights['w2'][expert].numpy(),cpu_array(weights['w2_scale'][expert]),n,k)).cuda()
        intermediate = x[tokens] @ w1
        activated = torch.nn.functional.silu(intermediate[:,:n])*intermediate[:,n:]
        values = (activated @ w2)*route[tokens,slots,None]
        output.index_add_(0,tokens,values)
    return output * (sample['routed_scaling_factor'] if sample['routed_scaling_factor'] is not None else 1.)


def dtype_reference(sample, weights):
    """Diagnostic BF16 boundaries; does not model split-K or atomic order."""
    import torch
    x=sample['hidden_states'].cuda().float()
    ids=sample['topk_ids'].cuda().long()
    route=sample['topk_weights'].cuda().bfloat16().float()
    k=x.shape[1];n=weights['w2'].shape[1]*16
    values=torch.zeros((*ids.shape,k),device=x.device,dtype=torch.float32)
    for expert in torch.unique(ids).tolist():
        tokens,slots=torch.where(ids==expert)
        w1=torch.from_numpy(dequantize(weights['w1'][expert].numpy(),cpu_array(weights['w1_scale'][expert]),k,2*n)).cuda().bfloat16().float()
        w2=torch.from_numpy(dequantize(weights['w2'][expert].numpy(),cpu_array(weights['w2_scale'][expert]),n,k)).cuda().bfloat16().float()
        first=(x[tokens]@w1).bfloat16().float()
        activated=(torch.nn.functional.silu(first[:,:n])*first[:,n:]).bfloat16().float()
        second=(activated@w2).bfloat16().float()
        values[tokens,slots]=(second*route[tokens,slots,None]).bfloat16().float()
    result=torch.zeros_like(x)
    for slot in range(ids.shape[1]):result+=values[:,slot]
    return (result*(sample['routed_scaling_factor'] or 1.)).bfloat16().float()


def repeatability_metrics(baseline, treatment, independent_control, fp32):
    """Fixed contrasts chosen before this run; envelope is descriptive only."""
    within_base=[errors(x,baseline[0]) for x in baseline[1:]]
    within_treatment=[errors(x,treatment[0]) for x in treatment[1:]]
    control=[errors(x,baseline[0]) for x in independent_control]
    drift=[errors(t,b) for b,t in zip(baseline,treatment)]
    pairs=within_base+control
    max_control=max(m['max_abs'] for m in pairs)
    rms_control=max(m['rms'] for m in pairs)
    return {'baseline_repeats':within_base,'treatment_repeats':within_treatment,
            'independent_control_graph':control,'paired_treatment_control':drift,
            'control_max_abs_envelope':max_control,'control_rms_envelope':rms_control,
            'max_cross_arm_exceeds_control_envelope':max(m['max_abs'] for m in drift)>max_control,
            'rms_cross_arm_exceeds_control_envelope':max(m['rms'] for m in drift)>rms_control,
            'baseline_vs_fp32':[errors(x,fp32) for x in baseline],
            'treatment_vs_fp32':[errors(x,fp32) for x in treatment],
            'limits':['finite repeat count is not a worst-case bound','comparison is descriptive and grants no numerical acceptance']}


def callable_for(sample, weights, override):
    import torch
    from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import fused_marlin_moe
    from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
    kwargs = {**weights,**sample,'use_atomic_add_override':override,'experiment_capture_tag':None}
    kwargs = {k: v.cuda() if isinstance(v,torch.Tensor) else v for k,v in kwargs.items()}
    def run():
        return fused_marlin_moe(**kwargs,workspace=marlin_make_workspace(kwargs['hidden_states'].device,max_blocks_per_sm=4))
    return run, kwargs


def capture_graph(fn):
    import torch
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(20): fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    before = torch.cuda.memory_allocated()
    with torch.cuda.graph(graph): output = fn()
    torch.cuda.synchronize()
    return graph, output, torch.cuda.memory_allocated()-before


def time_calls(fn, repeats=64):
    import torch
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(repeats): fn()
    end.record(); end.synchronize()
    return start.elapsed_time(end)/repeats


def timed_pairs(base, treatment, path):
    import torch
    for _ in range(20):
        base(); treatment()
    torch.cuda.synchronize()
    pairs = []
    with path.open('x') as stream:
        for i in range(30):
            order = ['baseline','treatment'] if i%2 == 0 else ['treatment','baseline']
            values = {}
            for arm in order: values[arm] = time_calls(base if arm=='baseline' else treatment)
            record = {'pair':i,'order':order,'repeats':64,'ms_per_call':values,'monotonic':time.monotonic()}
            stream.write(json.dumps(record)+'\n');stream.flush()
            pairs.append([values['baseline'],values['treatment']])
    return paired_summary(pairs)



@contextlib.contextmanager
def poison_actual_scratch():
    import importlib
    module=importlib.import_module('sglang.kernels.ops.moe.moe_wna16_marlin')
    original=module._jit_moe_wna16_marlin_module
    def poisoned_module(*args):
        native=original(*args)
        def gemm(*values):
            scratch=values[15]
            if scratch.numel(): scratch.fill_(float('nan'))
            return native.moe_wna16_marlin_gemm(*values)
        return types.SimpleNamespace(moe_wna16_marlin_gemm=gemm)
    module._jit_moe_wna16_marlin_module=poisoned_module
    try:yield
    finally:module._jit_moe_wna16_marlin_module=original


def check_graph_outputs(graphs, outputs, kwargs, weights, original_ref, case):
    import torch
    records = []
    # Change graph inputs between replays, including routes, to expose stale scratch.
    original = [{k: value[k].clone() for k in ['hidden_states','topk_ids','topk_weights']} for value in kwargs]
    for variant in ['original','changed','restored']:
        for graph, value, saved in zip(graphs, kwargs, original):
            for key in saved:
                new = saved[key]
                if variant == 'changed':
                    new = new*.5 if key != 'topk_ids' else (new+1)%weights['w1'].shape[0]
                value[key].copy_(new)
            graph.replay()
        torch.cuda.synchronize()
        reference = original_ref if variant != 'changed' else fp32_reference(kwargs[0],weights)
        arrays={'baseline':cpu_array(outputs[0]),'treatment':cpu_array(outputs[1]),'poisoned':cpu_array(outputs[2]),'fp32':cpu_array(reference)}
        np.savez(case/f'{variant}-outputs.npz',**arrays)
        repeats={'baseline':[],'treatment':[],'independent_control':[]}
        for _ in range(8):
            for name,index in [('baseline',0),('treatment',1),('independent_control',3)]:
                graphs[index].replay();torch.cuda.synchronize()
                repeats[name].append(cpu_array(outputs[index]).copy())
        repeated={k:np.stack(v) for k,v in repeats.items()}
        np.savez(case/f'{variant}-repeats.npz',**repeated)
        dtype=cpu_array(dtype_reference(kwargs[0],weights))
        np.savez(case/f'{variant}-dtype-reference.npz',dtype_reference=dtype)
        row={'variant':variant,**qualify_outputs(**arrays),
             'repeatability':repeatability_metrics(**repeated,fp32=arrays['fp32']),
             'dtype_reference':{arm:errors(arrays[arm],dtype) for arm in ['baseline','treatment']}}
        if not all(np.isfinite(a).all() for a in [*arrays.values(),*repeated.values(),dtype]):
            raise ValueError('Nonfinite output; abort diagnostic timings')
        if not row['poison_agreement']['finite'] or row['poison_agreement']['nrms']>.001 or row['poison_agreement']['nmax']>.01:
            raise ValueError('Scratch poison output differs; abort diagnostic timings')
        records.append(row)
    return records



def observed_calls(base, treatment):
    import torch
    import importlib
    module = importlib.import_module('sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe')
    original = module.moe_wna16_marlin_gemm
    receipts = []
    def observe(*args, **kw):
        sms = torch.cuda.get_device_properties(args[0].device).multi_processor_count
        scratch = min(kw['size_n']*args[10].size(0),sms*4*kw['moe_block_size']*256)
        if kw['moe_block_size']==8: scratch *= 2
        receipt = {key:kw[key] for key in ['use_atomic_add','use_fp32_reduce','size_m','size_n','size_k','moe_block_size','top_k','mul_topk_weights']}
        receipt['scratch_expected_bytes'] = scratch*4 if kw['use_fp32_reduce'] and not kw['use_atomic_add'] else 0
        receipts.append(receipt)
        return original(*args,**kw)
    module.moe_wna16_marlin_gemm = observe
    try:
        outputs = [base(),treatment()]
    finally:
        module.moe_wna16_marlin_gemm = original
    if len(receipts)!=4 or [r['use_atomic_add'] for r in receipts] != [True,True,False,False] or not all(r['use_fp32_reduce'] for r in receipts):
        raise ValueError('Actual GEMM reduction flags differ from declared contrast')
    return outputs,receipts


def run_case(record, capture, out):
    import torch
    case = out/Path(record['case']).stem;case.mkdir()
    sample = torch.load(capture/record['case'],map_location='cpu',weights_only=True)
    weights = torch.load(capture/record['weights'],map_location='cpu',weights_only=True)
    ref = fp32_reference(sample,weights)
    shared_sample = {k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in sample.items()}
    shared_weights = {k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in weights.items()}
    base, kb = callable_for(shared_sample,shared_weights,None)
    treatment, kt = callable_for(shared_sample,shared_weights,False)
    assert torch.cuda.get_device_capability()[0]>=9
    inputs_before = {k:fingerprint(kb[k]) for k in ['hidden_states','topk_ids','topk_weights']}
    (eager_base, eager_treatment), calls = observed_calls(base,treatment)
    torch.save({'baseline':eager_base.detach().cpu(),'treatment':eager_treatment.detach().cpu(),'fp32':ref.detach().cpu()},case/'eager-outputs.pt')
    eager_metrics = {'baseline':errors(cpu_array(eager_base),cpu_array(ref)), 'treatment':errors(cpu_array(eager_treatment),cpu_array(ref))}
    torch.cuda.reset_peak_memory_stats()
    gb, ob, mb = capture_graph(base)
    gt, ot, mt = capture_graph(treatment)
    second_base,ks=callable_for(shared_sample,shared_weights,None)
    gs,obs,ms=capture_graph(second_base)
    poison_fn, kp = callable_for(shared_sample,shared_weights,False)
    with poison_actual_scratch():
        gp, op, mp = capture_graph(poison_fn)
    numeric = check_graph_outputs([gb,gt,gp,gs],[ob,ot,op,obs],[kb,kt,kp,ks],weights,ref,case)
    np.savez(case/'eager-outputs.npz',baseline=cpu_array(eager_base),treatment=cpu_array(eager_treatment))
    eager_graph = [errors(cpu_array(ob),cpu_array(eager_base)),errors(cpu_array(ot),cpu_array(eager_treatment))]
    result = {'record':record,'actual_gemm_calls':calls,'eager_reference':eager_metrics,'numerical':numeric,'eager_graph':eager_graph,
              'graph_allocated_bytes':{'baseline':mb,'treatment':mt,'poison_qualification':mp,'second_control':ms},'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
              'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
    passed = all(row['passed'] for row in numeric) and all(m['finite'] and m['nrms']<=.001 and m['nmax']<=.01 for m in eager_graph)
    result['numerical_passed'] = passed
    torch.save({'baseline':ob.detach().cpu(),'treatment':ot.detach().cpu(),'poisoned_scratch':op.detach().cpu(),'fp32':ref.detach().cpu()},case/'outputs.pt')
    if not all(m['finite'] for m in eager_graph):
        raise ValueError('Nonfinite eager/graph output; abort diagnostic timings')
    result['graph_timing'] = timed_pairs(gb.replay,gt.replay,case/'graph-pairs.jsonl')
    result['eager_timing'] = timed_pairs(base,treatment,case/'eager-pairs.jsonl')
    result['control_graph_timing'] = timed_pairs(gb.replay,gs.replay,case/'control-graph-pairs.jsonl')
    result['timing_scope']='diagnostic even when v3 numerical gate rejects; no serving advancement'
    inputs_after = {k:fingerprint(kb[k]) for k in inputs_before}
    assert inputs_before==inputs_after
    assert inputs_after=={k:fingerprint(kt[k]) for k in inputs_before}
    assert inputs_after=={k:fingerprint(ks[k]) for k in inputs_before}
    assert inputs_after=={k:fingerprint(kp[k]) for k in inputs_before}
    result['inputs_before']=inputs_before;result['inputs_after']=inputs_after
    (case/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--capture',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(exist_ok=False)
    import torch
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    import sglang
    from sglang.srt.server_args import prepare_server_args, set_global_server_args_for_scheduler
    runtime_argv=json.loads((args.capture.parent/'server-args.json').read_text())
    if runtime_argv[1]!='serve':raise ValueError('Unexpected preserved serving command')
    from runtime_views import prepare_runtime_views
    prepare_runtime_views(runtime_argv,args.out.parent/'runtime-view.json')
    set_global_server_args_for_scheduler(prepare_server_args(runtime_argv[2:]))
    patch_manifest=json.loads((args.capture.parent/'patch-manifest.json').read_text())
    installed_root=Path(sglang.__file__).parent.parent
    for name,identity in patch_manifest.items():
        if hashlib.sha256((installed_root/name).read_bytes()).hexdigest()!=identity['after']:
            raise ValueError('Installed candidate source changed before replay: '+name)
    records=[json.loads(line) for line in (args.capture/'cases.jsonl').read_text().splitlines()]
    expected={f'layer-{layer:02d}-replay-{step:02d}.pt' for layer in [0,24,47] for step in [8,16,32]}
    assert len(records)==9 and {r['case'] for r in records}==expected
    for record in records:
        assert record['raw_bs']==10 and record['mode']=='TARGET_VERIFY'
        assert record['rows']==record['hidden_shape'][0]
        for key,hash_key in [('case','sha256'),('weights','weights_sha256')]:
            assert hashlib.sha256((args.capture/record[key]).read_bytes()).hexdigest()==record[hash_key]
    sentinel=verify_pack_sentinel()
    results=[]
    for record in records:
        result=run_case(record,args.capture,args.out);results.append(result)
        if sum(p.stat().st_size for p in args.out.rglob('*') if p.is_file())>1024**3:
            raise ValueError('Diagnostic output exceeds 1 GiB cap')
        # The original tolerance verdict is retained, but does not suppress diagnostics.
        torch.cuda.empty_cache()
    timing=[r['graph_timing'] for r in results if 'graph_timing' in r]
    eligible=(len(results)==9 and all(r['numerical_passed'] for r in results)
              and all(t['geometric_speedup']>=1.03 and t['bootstrap_95'][0]>1.01 and t['faster_pairs']>=27 for t in timing))
    result={'sentinel':sentinel,'runtime_args_sha256':hashlib.sha256((args.capture.parent/'server-args.json').read_bytes()).hexdigest(),'torch':torch.__version__,'device':torch.cuda.get_device_name(),
            'cases_completed':len(results),'advance_to_serving_preparation':False,'original_gate_and_performance_passed':eligible,'case_graph_summaries':timing,
            'limits':['one GPU/session; no between-run serving uncertainty','micro graphs do not certify full model graph residency','numerical tolerance is not gameplay or output-token equivalence']}
    (args.out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
