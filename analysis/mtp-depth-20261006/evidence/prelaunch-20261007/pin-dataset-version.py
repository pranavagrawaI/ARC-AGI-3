# Reproduce the frozen metadata pin after build_notebook.py, without altering its cells.
import json
from pathlib import Path
p = Path(__file__).resolve().parents[2] / 'kaggle/notebook/kernel-metadata.json'
m = json.loads(p.read_text())
m['dataset_sources'] = ['dfranzen/pennyroyal-v253', 'pranavagrawai/mtp-depth-replay-data-20261006/2']
p.write_text(json.dumps(m, indent=2) + '\n')
