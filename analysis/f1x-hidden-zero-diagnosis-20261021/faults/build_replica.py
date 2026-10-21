"""Rebuild the F1x Kaggle tree on CPU by executing the notebook's own patch/overlay/F1 cells against a local bundle copy.

usage: build_replica.py NOTEBOOK BUNDLE_SRC OUT_DIR
"""
import json, re, shutil, subprocess, sys
from pathlib import Path

nb_path, bundle_src, out = map(Path, sys.argv[1:4])
code = [''.join(c['source']) for c in json.loads(nb_path.read_text())['cells'] if c['cell_type'] == 'code']
patch_cell = next(c for c in code if c.startswith('%%writefile /kaggle/harness-changes.patch'))
overlay_cell = next(c for c in code if c.startswith('# M2 B1: install verified source'))
f1_cell = next((c for c in code if c.startswith('# M2 F1: preload')), None)

if out.exists(): shutil.rmtree(out)
shutil.copytree(bundle_src, out)
patch = out / 'harness-changes.patch'
patch.write_text(patch_cell.split('\n', 1)[1].rstrip('\n') + '\n')  # %%writefile ends the file with a newline
subprocess.run(['git', 'apply', '--include=ARC3-Inference/*', '-v', str(patch)], cwd=out / 'src', check=True,
               capture_output=True)
ns = {'BUNDLE_DIR': out}
exec(compile(overlay_cell, '<overlay-cell>', 'exec'), ns)
if f1_cell: exec(compile(f1_cell, '<f1-cell>', 'exec'), ns)
print(json.dumps({'replica': str(out), 'f1_applied': f1_cell is not None}))
