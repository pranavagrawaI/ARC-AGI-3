"""Read-only test of historical REAP mask against current captured routing.

This measures exposure to the proposed deletion, not pruned-model quality.
Run with the reference nooa venv Python (NumPy); no GPU or network required.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parent
sys.path.insert(0, str(ANALYSIS / 'marlin-numerics-diagnosis-20261010'))
from capture_io import load


def main():
    source = ANALYSIS / 'marlin-fixed-replay-r3-20261011/capture-source.json'
    manifest = json.loads(source.read_text())
    raw = ANALYSIS / 'marlin-reduction-replay-v3-20261010/readout/raw/marlin-replay'
    mask_path = Path('/home/pranav/projects/arc-agi-3-reference/duck-qwen38/runs/flash-next/prune/k448_mask_layers.json')
    mask = json.loads(mask_path.read_text())
    assert set(mask) == {str(i) for i in range(48)}
    assert all(len(v) == len(set(v)) == 448 and all(0 <= i < 512 for i in v) for v in mask.values())
    results = []
    hashes = {}
    for record in manifest['records']:
        name = 'capture/' + record['case']
        path = raw / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == manifest['files'][name] == record['sha256']
        hashes[name] = digest
        sample = load(path)
        ids = np.asarray(sample['topk_ids'])
        weights = np.asarray(sample['topk_weights'])
        assert ids.shape == weights.shape == tuple(record['route_shape'])
        assert np.isfinite(weights).all() and (weights >= 0).all() and weights.sum() > 0
        layer = str(int(path.name.split('-')[1]))
        removed = ~np.isin(ids, mask[layer])
        results.append(dict(case=record['case'], selected_routes=int(ids.size),
                            removed_selected_routes=int(removed.sum()),
                            removed_selected_weight_fraction=float(weights[removed].sum() / weights.sum())))
    report = dict(mask_sha256=hashlib.sha256(mask_path.read_bytes()).hexdigest(),
                  capture_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  verified_sample_hashes=hashes, cases=results,
                  scope='Historical expert IDs applied to unmodified current routing; no rerouting or pruning performed.',
                  limits=['Only three layers and nine captured batches.',
                          'Current checkpoint router/expert identity correspondence remains to be verified.',
                          'After pruning, top-k selection and normalization change; these are not pruned outputs.',
                          'No quality, throughput or GPU-memory qualification.'])
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
