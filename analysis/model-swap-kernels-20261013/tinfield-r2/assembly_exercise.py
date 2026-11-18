"""Run the r2 notebook's own assemble() against real saved indices.

Synthetic header-only shards carry the real tensor names (core: HF Tinfield index,
whose size matches the Kaggle core index byte-for-byte; base: real Intel PLE shard
header). The core shard set mirrors the live Kaggle core file listing, which lacks
model-00001-of-00010.safetensors. Usage: python -B assembly_exercise.py <notebook>
"""
import json, re, shutil, struct, sys, tempfile, traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
EV = ROOT/'tinfield/evidence'
KAGGLE_CORE_SHARDS = [f'model-{i:05d}-of-00010.safetensors' for i in range(2, 11)]  # live listing 2026-10-13

def write_shard(path, tensors):
    hdr = {}; pos = 0
    for name, t in tensors.items():
        hdr[name] = {'dtype': t.get('dtype', 'U8'), 'shape': t.get('shape', [2]), 'data_offsets': [pos, pos+2]}; pos += 2
    raw = json.dumps(hdr).encode(); raw += b' '*((-len(raw)) % 8)
    path.write_bytes(struct.pack('<Q', len(raw)) + raw + b'\0'*pos)

def main(nb_path):
    nb = json.loads(Path(nb_path).read_text())
    src = next(''.join(c['source']) for c in nb['cells'] if 'def assemble(' in ''.join(c['source']))
    start = src.index('"""Assemble public Tinfield'); end = src.index("\nMODEL_DIR = assemble(")
    tmp = Path(tempfile.mkdtemp(prefix='tinfield-asm-'))
    code = src[start:end].replace('/kaggle/working', str(tmp/'working'))
    (tmp/'working').mkdir()
    core, base = tmp/'core', tmp/'base'; core.mkdir(); base.mkdir()
    ci = json.loads((EV/'tin/model.safetensors.index.json').read_text())
    for f in ['config.json', 'model.safetensors.index.json', 'tokenizer.json']: shutil.copy(EV/'tin'/f, core/f)
    (core/'hf_quant_config.json').write_text('{}')
    by = {}
    for k, v in ci['weight_map'].items(): by.setdefault(v, {})[k] = {}
    for s in KAGGLE_CORE_SHARDS: write_shard(core/s, by[s])
    intel = ROOT/'azfocus/evidence/intel'
    for f in ['config.json', 'model.safetensors.index.json', 'tokenizer.json']: shutil.copy(intel/f, base/f)
    bi = json.loads((intel/'model.safetensors.index.json').read_text())['weight_map']
    for s in sorted({v for k, v in bi.items() if '.ngram_embedding.shard_' in k}):
        write_shard(base/s, json.loads((intel/(s+'.header.json')).read_text())['tensors'])
    ns = {}; exec(compile(code, 'notebook-cell-4-assembly', 'exec'), ns)
    report = {'notebook': str(nb_path), 'core_index_ple_named_keys': [k for k in ci['weight_map'] if '.ple_embedding.ngram_embedding' in k],
              'core_index_shards': sorted(set(ci['weight_map'].values())), 'kaggle_core_shards_present': KAGGLE_CORE_SHARDS}
    try:
        view = ns['assemble'](core, base, tmp/'view')
        idx = json.loads((Path(view)/'model.safetensors.index.json').read_text())['weight_map']
        missing = sorted({v for v in idx.values() if not (Path(view)/v).exists()})
        report.update(ok=not missing, tensors=len(idx), ple_shard_keys=sum('.shard_' in k for k in idx),
                      unsharded_ngram_in_final=[k for k in idx if k.endswith('ngram_embedding.weight')], missing_files=missing,
                      config_quant=json.loads((Path(view)/'config.json').read_text())['quantization_config'].get('quant_algo'))
    except Exception as e:
        report.update(ok=False, error=repr(e), traceback=traceback.format_exc()[-1500:])
    print(json.dumps(report, indent=2)); return report

if __name__ == '__main__':
    r = main(sys.argv[1]); sys.exit(0 if r['ok'] else 1)
