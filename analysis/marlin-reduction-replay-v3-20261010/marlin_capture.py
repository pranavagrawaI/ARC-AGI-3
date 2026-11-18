"""Explicit experiment hooks. Config absent means no binding or capture.

Requires patched exact wheel. Device copies are recorded inside capture graphs;
CPU serialization runs only after real, armed S10 target verification replays.
"""
import hashlib
import json
import os
from pathlib import Path

_ACTIVE = None
_BUFFERS = {}
_COUNTS = {}
CONFIG = None
WEIGHTS = ['w1', 'w2', 'w1_scale', 'w2_scale', 'g_idx1', 'g_idx2', 'sort_indices1', 'sort_indices2', 'w1_zeros', 'w2_zeros', 'w1_global_scale', 'w2_global_scale', 'w1_bias', 'w2_bias', 'expert_map']
INPUTS = ['hidden_states', 'gating_output', 'topk_weights', 'topk_ids']
SCALARS = ['global_num_experts', 'num_bits', 'is_k_full', 'inplace', 'routed_scaling_factor', 'clamp_limit', 'gemm1_alpha', 'activation', 'is_gated']


def config():
    global CONFIG
    name = os.environ.get('MARLIN_EXPERIMENT_CONFIG')
    if name is None:
        return None
    if CONFIG is None:
        CONFIG = json.loads(Path(name).read_text())
        if CONFIG['mode'] not in {'capture', 'control', 'treatment'}:
            raise ValueError('Unsupported experiment mode')
    return CONFIG


def target_kernels(runner):
    if runner.is_draft_worker:
        return []
    from sglang.srt.hardware_backend.gpu.quantization.gptq_kernels import GPTQMarlinMoEKernel
    return [(name, layer, layer.scheme.kernel) for name, layer in runner.model.named_modules()
            if isinstance(getattr(getattr(layer, 'scheme', None), 'kernel', None), GPTQMarlinMoEKernel)]


def bind_model(runner):
    cfg = config()
    if cfg is None:
        return
    kernels = target_kernels(runner)
    out = Path(cfg['out']); out.mkdir(parents=True, exist_ok=True)
    receipt = {'is_draft_worker': runner.is_draft_worker, 'mode': cfg['mode'], 'modules': [n for n, _, _ in kernels]}
    if not runner.is_draft_worker:
        if len(kernels) != cfg['expected_target_layers']:
            raise ValueError(f'Expected {cfg["expected_target_layers"]} target GPTQ MoE kernels, got {len(kernels)}')
        for ordinal, (name, layer, kernel) in enumerate(kernels):
            kernel.experiment_atomic_override = False if cfg['mode'] == 'treatment' else None
            if cfg['mode'] == 'capture' and ordinal in cfg['layer_ordinals']:
                kernel.experiment_capture_tag = f'layer-{ordinal:02d}'
                receipt.setdefault('selected', {})[kernel.experiment_capture_tag] = name
    role = 'draft' if runner.is_draft_worker else 'target'
    (out / f'binding-{role}.json').write_text(json.dumps(receipt, indent=2)+'\n')


def begin_capture(runner, key):
    global _ACTIVE
    cfg = config()
    _ACTIVE = None
    if cfg is not None and cfg['mode'] == 'capture' and not runner.model_runner.is_draft_worker:
        _ACTIVE = (id(runner.model_runner), repr(key))


def end_capture():
    global _ACTIVE
    _ACTIVE = None


def validate_call(args):
    import torch
    if args['num_bits'] != 4 or args['hidden_states'].dtype != torch.bfloat16:
        raise ValueError('Reference supports target W4/BF16 only')
    if any(args[key].dtype != torch.bfloat16 for key in ['w1_scale','w2_scale']) or any(args[key].dtype != torch.int32 for key in ['w1','w2']):
        raise ValueError('Reference requires integer W4 with BF16 scales')
    for key in ['expert_map','w1_zeros','w2_zeros','w1_global_scale','w2_global_scale','w1_bias','w2_bias']:
        if args[key] is not None:
            raise ValueError(f'Unsupported reference feature: {key}')
    for key in ['g_idx1','g_idx2','sort_indices1','sort_indices2']:
        if args[key] is not None and args[key].numel() != 0:
            raise ValueError(f'Unsupported reference feature: {key}')
    if not args['is_k_full'] or args['inplace'] or not args['is_gated'] or args['activation'] != 'silu' or args['clamp_limit'] is not None or args['gemm1_alpha'] is not None:
        raise ValueError('Reference supports full-K gated SiLU with no clamp/alpha')


def snapshot(tag, args):
    if _ACTIVE is None:
        return
    import torch
    validate_call(args)
    # Replace warmup buffers: the final call records these copy nodes into graph.
    buf = {key: torch.empty_like(args[key]) for key in INPUTS}
    for key in INPUTS:
        buf[key].copy_(args[key])
    _BUFFERS[(_ACTIVE, tag)] = (buf, {key: args[key] for key in WEIGHTS + SCALARS})


def save_exclusive(path, value):
    import torch
    with path.open('xb') as stream:
        torch.save(value, stream)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''): digest.update(chunk)
    return digest.hexdigest()


def after_execute(runner, forward_batch):
    cfg = config()
    if cfg is None or cfg['mode'] != 'capture' or runner.model_runner.is_draft_worker:
        return
    out = Path(cfg['out'])
    if not (out / 'ARM-CAPTURE').is_file() or forward_batch.batch_size != 10 or forward_batch.forward_mode.name != 'TARGET_VERIFY':
        return
    key = (id(runner.model_runner), repr(runner._replay_graph_key))
    count = _COUNTS.get(key, 0) + 1; _COUNTS[key] = count
    if count not in cfg['replay_ordinals']:
        return
    import torch
    torch.cuda.synchronize()
    records = []
    for tag in cfg['selected_tags']:
        buf, args = _BUFFERS[(key, tag)]  # missing snapshots invalidate, never substitute another graph
        weights_path = out / f'{tag}-weights.pt'
        if not weights_path.exists():
            required = sum(v.numel()*v.element_size() for k,v in args.items() if k in WEIGHTS and isinstance(v,torch.Tensor))
            if sum(p.stat().st_size for p in out.glob('*.pt')) + required + 65536 > cfg['max_capture_bytes']:
                raise RuntimeError('Capture weight storage cap would be exceeded')
            weights_hash = save_exclusive(weights_path, {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in args.items() if k in WEIGHTS})
            (out / f'{tag}-weights.sha256').write_text(weights_hash+'\n')
        case_path = out / f'{tag}-replay-{count:02d}.pt'
        sample = {k: v.detach().cpu() for k, v in buf.items()}
        sample.update({k: args[k] for k in SCALARS})
        sha = save_exclusive(case_path, sample)
        ids = sample['topk_ids'].flatten()
        experts, counts = torch.unique(ids, return_counts=True)
        record = {'case': case_path.name, 'sha256': sha, 'weights': weights_path.name,
                  'weights_sha256': (out / f'{tag}-weights.sha256').read_text().strip(),
                  'graph_key': key[1], 'raw_bs': forward_batch.batch_size, 'rows': runner.raw_num_token,
                  'hidden_shape': list(buf['hidden_states'].shape), 'route_shape': list(buf['topk_ids'].shape),
                  'mode': forward_batch.forward_mode.name, 'replay_ordinal': count,
                  'expert_counts': list(zip(experts.tolist(), counts.tolist()))}
        records.append(record)
    with (out / 'cases.jsonl').open('a') as stream:
        for record in records: stream.write(json.dumps(record)+'\n')
    if sum(p.stat().st_size for p in out.glob('*.pt')) > cfg['max_capture_bytes']:
        raise RuntimeError('Capture storage cap exceeded')
