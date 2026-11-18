"""Freeze an offline serving probe from the inspected donor assets."""
import base64
import hashlib
import json
import lzma
from pathlib import Path

HERE = Path(__file__).resolve().parent
DONOR = Path('/tmp/flash-next-keith-20260909')
def digest(data):
    return hashlib.sha256(data).hexdigest()

def build():
    corpus = json.loads((HERE.parent / 'prefix-cache-20260907/corpus.json').read_text())
    records = [corpus['batch_records'][0], corpus['pairs'][0]['successor'], *corpus['batch_records'][1:]]
    files = {'probe.py': (HERE / 'probe.py').read_bytes(),
             'actor-cases.json': json.dumps(records).encode(),
             'EXPERIMENT-CARD.md': (HERE / 'EXPERIMENT-CARD.md').read_bytes(),
             'donor-notebook.ipynb': (DONOR / 'duck-qwen3-8-flash-next-nvfp4-mtp.ipynb').read_bytes(),
             'donor-serving_setup.py': (DONOR / 'serving_setup.py').read_bytes()}
    manifest = {name: digest(data) for name, data in files.items()}
    source = 'import base64, hashlib, lzma, subprocess, sys\nfrom pathlib import Path\n'
    source += "root=Path('/kaggle/working'); root.mkdir(exist_ok=True)\n"
    for name, data in files.items():
        payload = base64.b64encode(lzma.compress(data)).decode()
        source += f"data=lzma.decompress(base64.b64decode({payload!r}))\nassert hashlib.sha256(data).hexdigest()=={digest(data)!r}\n(root/{name!r}).write_bytes(data)\n"
    source += "with (root/'probe.log').open('w') as log:\n    result=subprocess.run([sys.executable,str(root/'probe.py')],stdout=log,stderr=subprocess.STDOUT,timeout=2650)\nprint((root/'probe.log').read_text()[-16000:])\nassert result.returncode==0\n"
    compile(source, '<notebook>', 'exec')
    notebook = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'},
    }, 'cells': [
        {'id': 'probe-description', 'cell_type': 'markdown', 'metadata': {}, 'source': 'Flash-Next serving feasibility only. See embedded experiment card. No gameplay or promotion.'},
        {'id': 'probe-runner', 'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': source}]}
    metadata = json.loads((DONOR / 'kernel-metadata.json').read_text())
    for field in ['id_no', 'keywords']:
        metadata.pop(field, None)
    metadata.update(id='pranavagrawai/flash-next-feasibility-20260909', title='Flash Next Feasibility 20260909', is_private=True, code_file='probe.ipynb')
    for name, value in [('probe.ipynb', notebook), ('kernel-metadata.json', metadata), ('manifest.json', manifest)]:
        (HERE / name).write_text(json.dumps(value, indent=1)+'\n')
    print(json.dumps(manifest, indent=2))

if __name__ == '__main__':
    build()
