"""Temporal expert locality and cold-contribution replay on immutable S10 captures."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'
import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
sys.path.insert(0, str(PROJECT/'analysis/marlin-numerics-diagnosis-20261010'))
from capture_io import load
sys.path.insert(0, str(PROJECT/'analysis/marlin-fixed-replay-r3-20261011'))
from reference import dequantize, errors

CAPTURE = PROJECT/'analysis/marlin-reduction-replay-v3-20261010/readout/raw/marlin-replay/capture'
REFERENCES = PROJECT/'analysis/marlin-fixed-replay-r3-20261011/readout/raw/marlin-replay/benchmark'
SOURCE = PROJECT/'analysis/marlin-fixed-replay-r3-20261011/capture-source.json'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def bank(counts, size):
    # Fixed expert-ID tie break; unseen experts may fill larger banks.
    return np.lexsort((np.arange(len(counts)), -counts))[:size]


def coverage(ids, weights, kept):
    hits = np.isin(ids, kept)
    return {'route_fraction': float(hits.mean()),
            'distinct_expert_fraction': float(np.isin(np.unique(ids), kept).mean()),
            'route_mass_fraction': float(weights[hits].sum(dtype=np.float64)/weights.sum(dtype=np.float64))}


def reference_replay(sample, weights, kept):
    """Brute-force full and bank-only FP32 output, with identical arithmetic."""
    x = np.asarray(sample['hidden_states'], dtype=np.float32)
    ids = np.asarray(sample['topk_ids'])
    route = np.asarray(sample['topk_weights'], dtype=np.float32)
    hidden = x.shape[1]
    intermediate = weights['w2'].shape[1]*16
    values = np.zeros((*ids.shape, hidden), dtype=np.float32)
    start = time.monotonic()
    for index, expert in enumerate(np.unique(ids)):
        rows, slots = np.where(ids == expert)
        w1 = dequantize(weights['w1'][expert], weights['w1_scale'][expert], hidden, 2*intermediate)
        w2 = dequantize(weights['w2'][expert], weights['w2_scale'][expert], intermediate, hidden)
        z = x[rows] @ w1
        gate = z[:, :intermediate]
        # Stable SiLU is algebraically the saved reference's activation.
        exp = np.exp(-np.abs(gate))
        sigmoid = np.where(gate >= 0, 1/(1+exp), exp/(1+exp))
        activation = gate*sigmoid*z[:, intermediate:]
        values[rows, slots] = (activation @ w2)*route[rows, slots, None]
        if index % 64 == 0:
            print('expert', int(expert), 'elapsed', round(time.monotonic()-start, 1), flush=True)
    full = np.zeros_like(x)
    bank_only = np.zeros_like(x)
    hits = np.isin(ids, kept)
    # Saved GPU reference uses sequential FP32 index_add_ in expert-ID order.
    for expert in np.unique(ids):
        rows, slots = np.where(ids == expert)
        full[rows] += values[rows, slots]
        if expert in kept:
            bank_only[rows] += values[rows, slots]
    scale = sample.get('routed_scaling_factor') or 1.
    full *= scale
    bank_only *= scale
    return full, bank_only, values


def main():
    if (ROOT/'run-start.json').exists():
        raise ValueError('One run only; preserve existing evidence')
    started = time.monotonic()
    (ROOT/'run-start.json').write_text(json.dumps({'pid': os.getpid(), 'unix_time': time.time()})+'\n')
    source = json.loads(SOURCE.read_text())
    expected = source['files']
    hashes = {str(SOURCE): sha(SOURCE), str(ROOT/'probe.py'): sha(ROOT/'probe.py'),
              str(ROOT/'EXPERIMENT-CARD.md'): sha(ROOT/'EXPERIMENT-CARD.md')}
    records = source['records']
    samples = {}
    for record in records:
        path = CAPTURE/record['case']
        identity = sha(path)
        assert identity == record['sha256'] == expected['capture/'+record['case']]
        hashes[str(path)] = identity
        sample = load(path)
        ids, route = sample['topk_ids'], sample['topk_weights']
        assert ids.shape == route.shape == (40, 10)
        assert record['raw_bs'] == 10 and record['mode'] == 'TARGET_VERIFY'
        assert ids.min() >= 0 and ids.max() < 512
        assert np.isfinite(route).all() and (route >= 0).all() and (route.sum(axis=1) > 0).all()
        assert all(len(np.unique(row)) == 10 for row in ids)
        samples[(int(record['case'][6:8]), record['replay_ordinal'])] = sample
    rows = []
    rng = np.random.default_rng(20261011)
    with (ROOT/'routing.jsonl').open('x') as stream:
        for layer in [0, 24, 47]:
            training = samples[(layer, 8)]
            counts = np.bincount(training['topk_ids'].ravel(), minlength=512)
            for step in [16, 32]:
                sample = samples[(layer, step)]
                ids, route = sample['topk_ids'], sample['topk_weights']
                current_counts = np.bincount(ids.ravel(), minlength=512)
                for size in [128, 256, 448]:
                    kept = bank(counts, size)
                    observed = coverage(ids, route, kept)
                    oracle = coverage(ids, route, bank(current_counts, size))
                    random = [coverage(ids, route, rng.choice(512, size, replace=False))['route_fraction']
                              for _ in range(1000)]
                    row = {'layer': layer, 'train_step': 8, 'test_step': step, 'bank_size': size,
                           'kept': kept.tolist(), 'training_counts': counts.tolist(),
                           'test_counts': current_counts.tolist(), 'observed': observed, 'lookahead_oracle': oracle,
                           'random_mean': float(np.mean(random)), 'random_95': np.quantile(random, [.025, .975]).tolist()}
                    if step == 32:
                        cumulative = counts+np.bincount(samples[(layer, 16)]['topk_ids'].ravel(), minlength=512)
                        row['cumulative_8_16'] = coverage(ids, route, bank(cumulative, size))
                    stream.write(json.dumps(row, allow_nan=False)+'\n');stream.flush()
                    rows.append(row)
                    print('coverage', layer, step, size, observed['route_fraction'], flush=True)
    numeric = []
    layer_record = next(r for r in records if r['case'] == 'layer-00-replay-08.pt')
    weight_path = CAPTURE/layer_record['weights']
    assert sha(weight_path) == layer_record['weights_sha256']
    hashes[str(weight_path)] = layer_record['weights_sha256']
    weights = load(weight_path)
    kept = bank(np.bincount(samples[(0, 8)]['topk_ids'].ravel(), minlength=512), 256)
    for step in [16, 32]:
        case = f'layer-00-replay-{step:02d}'
        ref_path = REFERENCES/case/'original-outputs.npz'
        hashes[str(ref_path)] = sha(ref_path)
        reference = np.load(ref_path)['fp32']
        full, bank_only, values = reference_replay(samples[(0, step)], weights, kept)
        check = errors(full, reference)
        assert check['finite'] and check['nrms'] <= .0001, check
        influence = errors(bank_only, full)
        per_row = np.linalg.norm(bank_only-full, axis=1)/np.maximum(np.linalg.norm(full, axis=1), 1e-12)
        row = {'case': case, 'full_vs_saved_gpu_reference': check, 'omit_cold_vs_full': influence,
               'per_row_relative_l2': per_row.tolist(), 'median_relative_l2': float(np.median(per_row)),
               'maximum_relative_l2': float(per_row.max())}
        numeric.append(row)
        np.savez(ROOT/(case+'.npz'), full=full, bank_only=bank_only, per_expert_slot_contributions=values)
        print('output', json.dumps(row), flush=True)
    primary = [r for r in rows if r['bank_size'] == 256]
    result = {'routing_windows': 9, 'heldout_cases': len(primary),
              'primary_256_route_coverage': [r['observed']['route_fraction'] for r in primary],
              'passes_locality_screen': all(r['observed']['route_fraction'] >= .9 for r in primary),
              'layer_output_checks': numeric, 'cpu_seconds': time.monotonic()-started,
              'established_serving_speedup': False, 'established_arc_score_gain': False,
              'limits': ['Three nearby verification windows from one S10 run; correlated rows and drafts',
                         'Half-bank ranked by frequency, not REAP saliency',
                         'Omitting cold contributions does not recompute top-10 from the remaining router bank',
                         'A cache computes cold experts too; bank-only output is a deletion diagnostic',
                         'No CPU/GPU transfer, serving latency or game score measured']}
    (ROOT/'source-hashes.json').write_text(json.dumps(hashes, indent=2)+'\n')
    (ROOT/'RESULT.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print('DONE', json.dumps({k: v for k, v in result.items() if k != 'layer_output_checks'}), flush=True)


if __name__ == '__main__':
    main()
