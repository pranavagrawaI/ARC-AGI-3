"""CPU arithmetic probes against immutable Marlin v3 capture and GPU outputs.

This models dtype boundaries, not CUDA split-K/atomic accumulation order.
Never rewrites the experiment, gates, capture, or run verdict.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('OMP_NUM_THREADS', '4')
import numpy as np
from capture_io import load

ROOT = Path(__file__).resolve().parent
RUN = ROOT.parent / 'marlin-reduction-replay-v3-20261010'
sys.path.insert(0, str(RUN))
from reference import dequantize, errors


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def array(t):
    return np.asarray(t)


def bf16(x):
    """Round finite FP32 to nearest-even BF16, returned as FP32."""
    a = np.asarray(x, dtype=np.float32)
    if not np.isfinite(a).all():
        raise ValueError('Nonfinite probe input')
    u = a.view(np.uint32)
    rounded = (u + np.uint32(0x7fff) + ((u >> 16) & 1)) & np.uint32(0xffff0000)
    return rounded.view(np.float32)


def silu(x):
    # Stable formula, with exp evaluated in the selected FP32/FP64 dtype.
    s = np.exp(-np.abs(x))
    return x * np.where(x >= 0, 1 / (1 + s), s / (1 + s))


MODELS = {
    'fp64_math': (-1, np.float64),
    'fp32_math': (-1, np.float32),
    'output_rounding_only': (0, np.float32),
    'bf16_weights': (1, np.float32),
    'plus_gemm1_store': (2, np.float32),
    'plus_activation_store': (3, np.float32),
    'plus_gemm2_store': (4, np.float32),
    'plus_route_cast': (5, np.float32),
    'plus_route_product_store': (6, np.float32),
}


def replay(sample, weights, rows, variant):
    k = sample['hidden_states'].shape[1]
    n = weights['w2'].shape[1] * 16
    ids = array(sample['topk_ids'])[rows].copy()
    x = array(sample['hidden_states'])[rows].copy()
    route = array(sample['topk_weights'])[rows].copy()
    if variant == 'changed':
        x = bf16(x * .5)
        route *= .5
        ids = (ids + 1) % weights['w1'].shape[0]
    scale = sample['routed_scaling_factor'] or 1.
    outputs = {name: np.zeros((len(rows), ids.shape[1], k), dtype=dtype)
               for name, (_, dtype) in MODELS.items()}
    experts = []
    for expert in np.unique(ids):
        tokens, slots = np.where(ids == expert)
        w1 = dequantize(array(weights['w1'][expert]), array(weights['w1_scale'][expert]), k, 2*n)
        w2 = dequantize(array(weights['w2'][expert]), array(weights['w2_scale'][expert]), n, k)
        w1b, w2b = bf16(w1), bf16(w2)
        zfp32 = x[tokens] @ w1
        zbf16w = x[tokens] @ w1b
        zfp64 = x[tokens].astype(np.float64) @ w1.astype(np.float64)
        detail = {'expert': int(expert), 'tokens': [int(rows[t]) for t in tokens], 'slots': slots.tolist(), 'values_at_268': {}}
        for name, (stage, dtype) in MODELS.items():
            z = zfp64 if dtype == np.float64 else (zfp32 if stage < 1 else zbf16w)
            if stage >= 2:
                z = bf16(z)
            act = silu(z[:, :n]) * z[:, n:]
            if stage >= 3:
                act = bf16(act)
            w = w2.astype(np.float64) if dtype == np.float64 else (w2 if stage < 1 else w2b)
            down = act @ w
            if stage >= 4:
                down = bf16(down)
            r = route[tokens, slots].astype(dtype)
            if stage >= 5:
                r = bf16(r)
            values = down * r[:, None]
            if stage >= 6:
                values = bf16(values)
            outputs[name][tokens, slots] = values
            detail['values_at_268'][name] = values[:, 268].tolist()
        experts.append(detail)
    # moe_sum_reduce uses FP32 accumulation and BF16 final storage.
    final = {}
    for name, (stage, _) in MODELS.items():
        total = np.zeros((len(rows), k), dtype=outputs[name].dtype)
        for slot in range(ids.shape[1]):
            total += outputs[name][:, slot]
        total *= scale
        final[name] = bf16(total) if stage >= 0 else total
    return final, experts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rows', type=int, nargs='+', default=[7, 18, 22])
    parser.add_argument('--all-rows', action='store_true')
    parser.add_argument('--out', type=Path, default=ROOT / 'probe.json')
    args = parser.parse_args()
    start = time.monotonic()
    capture = RUN / 'readout/raw/marlin-replay/capture'
    case = RUN / 'readout/raw/marlin-replay/benchmark/layer-00-replay-08'
    record = next(json.loads(l) for l in (capture / 'cases.jsonl').read_text().splitlines()
                  if json.loads(l)['case'] == 'layer-00-replay-08.pt')
    paths = {'sample': capture / record['case'], 'weights': capture / record['weights']}
    hashes = {key: sha(path) for key, path in paths.items()}
    assert hashes['sample'] == record['sha256']
    assert hashes['weights'] == record['weights_sha256']
    sample = load(paths['sample'])
    weights = load(paths['weights'])
    if args.all_rows:
        args.rows = list(range(sample['hidden_states'].shape[0]))
    # Independent saved NPZ exports verify archive decoding for BF16 and FP32.
    tensor_outputs = load(case/'outputs.pt')
    exported = np.load(case/'restored-outputs.npz')
    for key in ['baseline', 'treatment', 'fp32']:
        np.testing.assert_array_equal(tensor_outputs[key], exported[key])
    # Known ties demonstrate nearest-even rather than truncation or ties-up.
    np.testing.assert_array_equal(bf16(np.array([1+1/256,1+3/256,-1-1/256],np.float32)),
                                  np.array([1,1+1/64,-1],np.float32))
    report = {'hashes': hashes, 'rows': args.rows, 'numpy': np.__version__, 'python': sys.version,
              'models': {key: {'rounding_stage': stage, 'math_dtype': np.dtype(dtype).name} for key, (stage,dtype) in MODELS.items()},
              'sample': {k: {'shape': list(v.shape), 'decoded_dtype': str(v.dtype)} if isinstance(v,np.ndarray) else v for k,v in sample.items()},
              'variants': {},
              'limits': ['CPU matmul does not model CUDA split-K/atomic order', 'dtype model is diagnostic, not a replacement acceptance oracle', 'final sum model assumes FP32 slot accumulation and BF16 output', 'archive BF16 tensors decode to FP32 for CPU arithmetic'],
              'full_rows': args.all_rows}
    for variant in ['original', 'changed']:
        print('replay', variant, flush=True)
        saved = np.load(case / f'{variant}-outputs.npz')
        predictions, experts = replay(sample, weights, args.rows, variant)
        np.savez(args.out.with_name(f'{args.out.stem}-{variant}.npz'), **predictions)
        detail = {'models': {}, 'expert_contributions': experts}
        for name, values in predictions.items():
            detail['models'][name] = {
                'vs_saved_fp32': errors(values, saved['fp32'][args.rows]),
                'vs_baseline': errors(values, saved['baseline'][args.rows]),
                'vs_treatment': errors(values, saved['treatment'][args.rows]),
                'at_18_268': float(values[args.rows.index(18), 268]) if 18 in args.rows else None,
                'at_22_268': float(values[args.rows.index(22), 268]) if 22 in args.rows else None}
        for key in ['baseline', 'treatment', 'fp32']:
            detail[key+'_at_18_268'] = float(saved[key][18,268])
            detail[key+'_at_22_268'] = float(saved[key][22,268])
        report['variants'][variant] = detail
        print(variant, json.dumps({k: v['at_18_268'] for k,v in detail['models'].items()}), flush=True)
    report['seconds'] = time.monotonic() - start
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print('saved', args.out, 'seconds', report['seconds'], flush=True)


if __name__ == '__main__':
    main()
