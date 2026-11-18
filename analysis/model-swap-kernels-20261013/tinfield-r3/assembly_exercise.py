"""Run the r3 notebook's own assemble() on a tree mirroring the CURRENT Kaggle mounts.

Inputs (kaggle-mounted/, fetched by fetch_mounted.py/fetch_headers.py on 2026-10-13):
live file listings, the real config/index/tokenizer of both instances, and the real
safetensors headers of every core shard and every base PLE shard (ranged reads).
Each synthetic shard carries its real header (names, dtype, shape) with 2-byte payloads.
"""
import json, shutil, struct, sys, tempfile, traceback
from pathlib import Path
HERE = Path(__file__).resolve().parent; M = HERE/'kaggle-mounted'

def write_shard(path, tensors):
    hdr = {}; pos = 0
    for name, t in tensors.items():
        if name == '__metadata__': hdr[name] = t; continue
        hdr[name] = {'dtype': t['dtype'], 'shape': t['shape'], 'data_offsets': [pos, pos+2]}; pos += 2
    raw = json.dumps(hdr).encode(); raw += b' '*((-len(raw)) % 8)
    path.write_bytes(struct.pack('<Q', len(raw)) + raw + b'\0'*pos)

def build(which, root):
    files = json.loads((M/which/'files.json').read_text())
    for f in files:
        p = root/f['name']; p.parent.mkdir(parents=True, exist_ok=True)
        real = M/which/f['name']; hdr = M/which/'headers'/(f['name']+'.json')
        if real.is_file() and f['name'] in ('config.json','model.safetensors.index.json','tokenizer.json'): shutil.copy(real, p)
        elif hdr.is_file(): write_shard(p, json.loads(hdr.read_text())['tensors'])
        elif f['name'].endswith('.safetensors'): write_shard(p, {'placeholder': {'dtype':'U8','shape':[2]}})
        else: p.write_text('{}')
    return files

def main(nb_path):
    nb = json.loads(Path(nb_path).read_text())
    src = next(''.join(c['source']) for c in nb['cells'] if 'def assemble(' in ''.join(c['source']))
    start = src.index('"""Assemble public Tinfield'); end = src.index("\nMODEL_DIR = assemble(")
    tmp = Path(tempfile.mkdtemp(prefix='tinfield-r3-asm-')); (tmp/'working').mkdir()
    code = src[start:end].replace('/kaggle/working', str(tmp/'working'))
    core, base = tmp/'core', tmp/'base'; build('core', core); build('base', base)
    ci = json.loads((core/'model.safetensors.index.json').read_text())['weight_map']
    core_hdr = {}
    for h in (M/'core/headers').glob('*.json'):
        for k in json.loads(h.read_text())['tensors']:
            if k != '__metadata__': core_hdr[k] = h.name.removesuffix('.json')
    report = {'notebook': str(nb_path), 'core_index_keys': len(ci), 'core_header_keys': len(core_hdr),
              'core_index_keys_without_mounted_file': sorted(k for k,v in ci.items() if not (core/v).exists()),
              'core_index_vs_header_mismatch': sorted(k for k,v in ci.items() if (core/v).exists() and core_hdr.get(k)!=v)[:20],
              'core_header_keys_not_indexed': sorted(set(core_hdr)-set(ci))[:20]}
    ns = {}; exec(compile(code, 'notebook-cell-4-assembly', 'exec'), ns)
    try:
        view = Path(ns['assemble'](core, base, tmp/'view'))
        idx = json.loads((view/'model.safetensors.index.json').read_text())['weight_map']
        missing = sorted({v for v in idx.values() if not (view/v).exists()})
        unbacked = []
        for shard in sorted(set(idx.values())):
            n = struct.unpack('<Q', (view/shard).read_bytes()[:8])[0]
            names = set(json.loads((view/shard).read_bytes()[8:8+n])) - {'__metadata__'}
            unbacked += [k for k,v in idx.items() if v==shard and k not in names]
        receipt = json.loads((tmp/'working/model-swap-assembly.json').read_text())
        report.update(ok=not missing and not unbacked and not report['core_index_vs_header_mismatch'],
                      tensors=len(idx), ple_shard_keys=sum('.ngram_embedding.shard_' in k for k in idx),
                      unsharded_ngram_in_final=[k for k in idx if k.endswith('ngram_embedding.weight')],
                      missing_files=missing, index_keys_not_in_their_shard_header=unbacked[:20],
                      ple_methods=sorted({s['method'] for s in receipt['ple_shards']}), ple_source_shards=len(receipt['ple_shards']),
                      linked_sidecars=sorted(p.name for p in view.iterdir() if p.is_symlink() and not p.name.endswith('.safetensors')),
                      config_quant=json.loads((view/'config.json').read_text())['quantization_config'].get('quant_algo'))
    except Exception as e:
        report.update(ok=False, error=repr(e), traceback=traceback.format_exc()[-1500:])
    print(json.dumps(report, indent=2)); return report

if __name__ == '__main__':
    sys.exit(0 if main(sys.argv[1])['ok'] else 1)
