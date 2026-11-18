"""Run the notebook's own routing_view() on a tree mirroring the CURRENT Kaggle mount of the incumbent.

Inputs: kaggle-mounted/base (live file listing + real config.json/index fetched by fetch_mounted.py).
Shards are empty placeholders with the mounted names; the real config and index are used.
Also runs cell 12's prepare_draft_view() (unchanged from c14) against the real drafter config/index,
with the view as target, and checks what the precache walk and the tokenizer/chat-template links see.
"""
import json
import os
import re
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
M = HERE / 'kaggle-mounted'


def mirror(which, root):
    files = json.loads((M / which / 'files.json').read_text())
    for f in files:
        p = root / f['name']
        p.parent.mkdir(parents=True, exist_ok=True)
        real = M / which / f['name']
        if real.is_file():
            shutil.copy(real, p)
        else:
            p.write_bytes(b'\0' * 8)
    return files


def cell(nb, marker):
    hits = [''.join(c['source']) for c in nb['cells'] if marker in ''.join(c['source'])]
    assert len(hits) == 1, marker
    return hits[0]


def main(nb_path):
    nb = json.loads(Path(nb_path).read_text())
    tmp = Path(tempfile.mkdtemp(prefix='a5b-asm-'))
    base, draft, working = tmp / 'base', tmp / 'draft', tmp / 'working'
    working.mkdir()
    base_files = mirror('base', base)
    mirror('draft', draft)
    c4 = cell(nb, 'def routing_view(')
    code = c4[c4.index("INCUMBENT_MODEL_DIR = "):c4.index("\nMODEL_DIR = routing_view(")]
    code = code.replace('/kaggle/working', str(working))
    ns = {'json': json, 'Path': Path}
    report = {'notebook': str(nb_path), 'mounted_files': len(base_files)}
    try:
        exec(compile(code, 'cell-4-routing-view', 'exec'), ns)
        view = Path(ns['routing_view'](base, tmp / 'view', ns['A5B_TOP_K']))
        cfg = json.loads((view / 'config.json').read_text())
        src = json.loads((base / 'config.json').read_text())
        index = json.loads((view / 'model.safetensors.index.json').read_text())
        walk = sorted(os.path.relpath(os.path.join(r, f), view) for r, _, fs in os.walk(view) for f in fs if os.path.isfile(os.path.join(r, f)))
        c12 = cell(nb, 'def prepare_draft_view(')
        fn = c12[c12.index('def prepare_draft_view('):c12.index('\n# ---- validate notebook inputs')]
        ns12 = {}
        exec('import copy, hashlib, json\nfrom pathlib import Path\n' + c12[c12.index('def find_unique('):c12.index('def prepare_draft_view(')]
             + 'EXPERT_TARGET = r"re:^mtp\\.layers\\.0\\.mlp\\.experts\\.[0-9]+\\.(gate_proj|up_proj|down_proj)$"\n'
             + 'DENSE_IGNORE = r"re:^(?!mtp\\.layers\\.0\\.mlp\\.experts(?:\\.|$)).*"\n' + fn, ns12)
        dview = Path(ns12['prepare_draft_view'](view, draft, tmp / 'prefix'))
        dcfg = json.loads((dview / 'config.json').read_text())
        q = cfg['quantization_config']
        report.update(
            ok=True,
            view_top_k=cfg['text_config']['num_experts_per_tok'], source_top_k=src['text_config']['num_experts_per_tok'],
            num_experts=cfg['text_config']['num_experts'],
            only_change=sorted(k for k in set(cfg['text_config']) | set(src['text_config']) if cfg['text_config'].get(k) != src['text_config'].get(k)),
            quant_method_seen_by_cell12_assert=(q.get('quant_method'), q.get('bits')),
            indexed_shards_missing_in_view=sorted({s for s in index['weight_map'].values() if not (view / s).is_file()}),
            precache_walk_files=len(walk), mounted_files_minus_config=len(base_files) - 1 + 1,
            tokenizer_and_template_present=[(n, (view / n).is_file()) for n in ('tokenizer.json', 'chat_template.jinja', 'quantization_config.json')],
            draft_view_top_k=dcfg['text_config']['num_experts_per_tok'], draft_num_experts=dcfg['text_config']['num_experts'],
            draft_links_target_tokenizer=os.path.realpath(dview / 'tokenizer.json') == os.path.realpath(base / 'tokenizer.json'),
            receipt=json.loads((working / 'model-swap-assembly.json').read_text()))
        report['ok'] = (report['view_top_k'] == 5 and report['source_top_k'] == 10 and report['only_change'] == ['num_experts_per_tok']
                        and not report['indexed_shards_missing_in_view'] and report['draft_view_top_k'] == 10
                        and report['precache_walk_files'] == len(base_files) and report['quant_method_seen_by_cell12_assert'] == ('auto-round', 4)
                        and report['draft_links_target_tokenizer'])
    except Exception as e:
        report.update(ok=False, error=repr(e), traceback=traceback.format_exc()[-2000:])
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    sys.exit(0 if main(sys.argv[1])['ok'] else 1)
