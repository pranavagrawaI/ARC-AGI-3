"""Patch only the exact packaged loader, then construct a target metadata view."""
import argparse
import hashlib
import json
import sysconfig
from pathlib import Path

from pruning import make_view

HERE = Path(__file__).resolve().parent
HASHES = {
    'qwen4_exp.py': '35a1785ce4c287821f00f1518b3880ab6838ef58d42fd51ece86d8098b2ed666',
    'qwen4_exp_mtp.py': 'e5d1f10bc66d73d4f52c0a139473ac34b6471481f239bfa7d498578a950ce3fb',
}


def patch_sources(root):
    edits = {
        'qwen4_exp.py': ('    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):\n',
                         '        from sglang.srt.models.retained_pruning import filter_target_weights\n        weights = filter_target_weights(self, weights)\n'),
        'qwen4_exp_mtp.py': ('        self.logits_processor = LogitsProcessor(config)\n',
                             '        from sglang.srt.models.retained_pruning import record_draft\n        record_draft(self)\n'),
    }
    outputs = {}
    for name, (anchor, addition) in edits.items():
        original = (root / name).read_bytes()
        if hashlib.sha256(original).hexdigest() != HASHES[name]:
            raise ValueError('Packaged model source changed: ' + name)
        text = original.decode()
        if text.count(anchor) != 1:
            raise ValueError('Loader patch anchor is not unique')
        outputs[name] = text.replace(anchor, anchor + addition)
        compile(outputs[name], name, 'exec')
    for name, text in outputs.items():
        (root / name).write_text(text)
    helper = (HERE / 'pruning.py').read_bytes()
    (root / 'retained_pruning.py').write_bytes(helper)
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in [*outputs, 'retained_pruning.py']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--view', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--mode', choices=['pruned', 'control'], required=True)
    args = parser.parse_args()
    expected = json.loads((HERE / 'contract.json').read_text())
    digest = hashlib.sha256((args.source / 'config.json').read_bytes()).hexdigest()
    if digest != expected['config_sha256']:
        raise ValueError('Mounted target checkpoint config differs')
    index_digest = hashlib.sha256((args.source / 'model.safetensors.index.json').read_bytes()).hexdigest()
    index = json.loads((args.source / 'model.safetensors.index.json').read_text())
    names_digest = hashlib.sha256(json.dumps(sorted(index['weight_map'])).encode()).hexdigest()
    if names_digest != expected['index_tensor_names_sha256']:
        raise ValueError('Mounted target tensor inventory differs')
    root = Path(sysconfig.get_paths()['purelib']) / 'sglang/srt/models'
    receipt = {'mode': args.mode, 'source_config_sha256': digest, 'source_index_sha256': index_digest,
               'patched_files': patch_sources(root)}
    if args.mode == 'pruned':
        make_view(args.source, args.view, json.loads((HERE / 'mask.json').read_text()),
                  json.loads((HERE / 'router-identity.json').read_text()))
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + '\n')
    print('RETAINED_PRUNING_INSTALL', json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
