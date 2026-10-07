"""Build a corrected wrapper around the immutable first sweep notebook."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent / 'mtp-depth-20261006'
RUN_ID = 'mtp-depth-retry1-20261007'
SLUG = 'mtp-depth-sweep-retry1-20261007'
TAG = 'mtp-depth-retry1-prelaunch-20261007'
COMPETITION = 'arc-prize-2026-arc-agi-3'


def build():
    notebook = json.loads((ORIGINAL / 'kaggle/notebook/mtp-depth-sweep-20261006.ipynb').read_text())
    guard = (HERE / 'hardware_gate.py').read_text()
    notebook['cells'].insert(1, {'cell_type': 'code', 'metadata': {}, 'execution_count': None,
                                 'outputs': [], 'source': guard.splitlines(keepends=True)})
    cell = notebook['cells'][-1]
    source = ''.join(cell['source'])
    marker = "print('provenance:', PROVENANCE)"
    assert source.count(marker) == 1
    source = source.replace(marker, f"PROVENANCE.update(run_id='{RUN_ID}', retry_prelaunch_tag='{TAG}')\n" + marker)
    cell['source'] = source.splitlines(keepends=True)
    notebook['cells'][0]['source'] = [f'# MTP corrected retry {RUN_ID}\n',
                                    'Original replay/decision rule retained; see EXPERIMENT-CARD.md.\n']
    metadata = json.loads((ORIGINAL / 'kaggle/notebook/kernel-metadata.json').read_text())
    metadata.update(id=f'pranavagrawai/{SLUG}', title=SLUG, code_file=f'{SLUG}.ipynb',
                    competition_sources=[COMPETITION])
    out = HERE / 'kernel'
    out.mkdir(exist_ok=True)
    (out / metadata['code_file']).write_text(json.dumps(notebook, indent=1) + '\n')
    (out / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    build()
