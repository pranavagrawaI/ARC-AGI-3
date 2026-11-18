"""CPU-only checks of the archived definitions, not deployed actor qualification."""
import ast
from collections import deque
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT/'experiments/agents/0158-nooa-tycho-capability-build/third_party/source-snapshots/tycho/wmlib_template.py'
names = {'segment', 'segment_summary', '_runs', 'plan_bfs_with_diagnostics'}
tree = ast.parse(SOURCE.read_text())
selected = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[])
ns = {'np': np, 'deque': deque}
exec(compile(selected, str(SOURCE), 'exec'), ns)
grid = np.zeros((8, 12), dtype=int)
grid[2:4, 7:10] = 3
parts = ns['segment'](grid, background=0, include_cells=True)
summary = ns['segment_summary'](grid, background=0)
assert parts[0]['bbox'] == (2, 7, 3, 9)
assert 'bbox=(2,7)-(3,9)' in summary
assert {(r,c) for r,c in parts[0]['cells']} == {(r,c) for r in range(2,4) for c in range(7,10)}
def bad_transition(model, state, action):
    raise ValueError('deliberately unknown transition')
ns.update(_target=lambda model,target: target, _planner_key=lambda model,state: state,
          _is_target=lambda target,state: target(state), _should_prune=lambda model,state: False,
          _acts=lambda model,state: ['right'], _planner_successor=bad_transition)
search = ns['plan_bfs_with_diagnostics'](object(), 0, lambda state: state == 1)
assert search['status'] == 'no_plan_exhausted' and search['errors']
manifest = ROOT/'deployment/kaggle/q6-summary-boundary-20260906/CHASSIS.json'
data = json.loads(manifest.read_text())
verified = {name: hashlib.sha256((manifest.parent/name).read_bytes()).hexdigest() == digest
            for name,digest in data['files'].items()}
assert all(verified.values())
result = {'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
          'base_contract_sha256': data['contract_sha256'], 'manifest_files_verified': verified,
          'segmentation': {'actual_row_bounds': [2,3], 'actual_col_bounds': [7,9],
                           'stored_bbox': parts[0]['bbox'], 'rendered_summary': summary,
                           'finding': 'Exact cells and r0,c0,r1,c1 summary agree; suspected coordinate defect rejected'},
          'failed_transition_search': search,
          'limits': 'AST-extracted archived definitions with explicit planner boundary stubs; no live actor or model execution'}
Path(__file__).with_name('EVIDENCE.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
