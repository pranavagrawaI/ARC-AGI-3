"""Target-only REAP pruning for the hash-bound pennyroyal Qwen4 loader.

The source checkpoint is immutable. A metadata view sets target experts to 448;
this iterator aligns router rows and individual GPTQ expert tensor names.
"""
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path

EXPERT = re.compile(r'^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(qweight|qzeros|scales)$')
ROUTER = re.compile(r'^model\.language_model\.layers\.(\d+)\.mlp\.gate\.weight$')


def tensor_bytes(tensor):
    if type(tensor).__module__.startswith('numpy'):
        return tensor.tobytes(order='C')
    import torch
    return tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()


def select_rows(tensor, rows):
    result = tensor[rows]
    return result.contiguous() if hasattr(result, 'contiguous') else result.copy()


def remap_name(name, maps):
    match = EXPERT.fullmatch(name)
    if match is None:
        if '.mlp.experts.' in name and not name.startswith('mtp.'):
            raise ValueError('Unsupported target expert tensor: ' + name)
        return name
    layer, old = int(match[1]), int(match[2])
    if layer not in maps or not 0 <= old < 512:
        raise ValueError('Expert outside qualified checkpoint')
    new = maps[layer].get(old)
    return None if new is None else name[:match.start(2)] + str(new) + name[match.end(2):]


def filtered(weights, mask, router_hashes, stats):
    maps = {int(layer): {old: new for new, old in enumerate(keep)} for layer, keep in mask.items()}
    seen = set()
    for name, tensor in weights:
        mapped = remap_name(name, maps)
        match = EXPERT.fullmatch(name)
        router = ROUTER.fullmatch(name)
        if match or router:
            if name in seen:
                raise ValueError('Duplicate target tensor: ' + name)
            seen.add(name)
        if match:
            stats['dropped' if mapped is None else 'kept'] += 1
        if mapped is None:
            continue
        if router:
            layer = router[1]
            if tuple(tensor.shape) != (512, 2560) or hashlib.sha256(tensor_bytes(tensor)).hexdigest() != router_hashes[layer]:
                raise ValueError('Router identity mismatch: ' + name)
            tensor = select_rows(tensor, mask[layer])
            stats['routers'] += 1
        yield mapped, tensor
    if dict(stats) != {'kept': 48 * 448 * 9, 'dropped': 48 * 64 * 9, 'routers': 48}:
        raise ValueError('Incomplete pruning stream: ' + repr(dict(stats)))


def expert_shapes(model):
    return {name: list(tensor.shape) for name, tensor in model.named_parameters()
            if '.mlp.experts.' in name and tensor.ndim >= 3}


def filter_target_weights(model, weights):
    spec = getattr(model.config, 'retained_pruning', None)
    if spec is None:
        yield from weights
        directory = os.environ.get('RETAINED_PRUNING_RECEIPT_DIR')
        if directory:
            shapes = expert_shapes(model)
            layers = {int(name.split('.layers.')[1].split('.')[0]) for name in shapes}
            if model.config.num_experts != 512 or layers != set(range(48)) or any(s[0] != 512 for s in shapes.values()):
                raise ValueError('Control expert banks differ')
            destination = Path(directory) / 'target.json'
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps({'complete': True, 'target_experts': 512, 'shapes': shapes}, indent=2) + '\n')
        return
    if model.config.num_experts != 448:
        raise ValueError('Pruned target configuration differs')
    stats = Counter(kept=0, dropped=0, routers=0)
    receipt = {'complete': False, 'mask_sha256': spec['mask_sha256']}
    destination = Path(os.environ['RETAINED_PRUNING_RECEIPT_DIR']) / 'target.json'
    try:
        yield from filtered(weights, spec['mask'], spec['router_hashes'], stats)
        shapes = expert_shapes(model)
        layers = {int(name.split('.layers.')[1].split('.')[0]) for name in shapes}
        if layers != set(range(48)) or any(shape[0] != 448 for shape in shapes.values()):
            raise ValueError('Live target expert banks differ')
        receipt.update(complete=True, shapes=shapes, target_experts=448)
    finally:
        receipt['counts'] = dict(stats)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(receipt, indent=2) + '\n')
        print('RETAINED_PRUNING_TARGET', json.dumps({k: v for k, v in receipt.items() if k != 'shapes'}), flush=True)


DRAFT_EXPERTS = 512


def is_identity_layout(metadata):
    """True when every logical expert lives in the physical slot of the same id on one rank."""
    import torch
    n = metadata.num_logical_experts
    p2l = metadata.physical_to_logical_map_cpu
    l2p = metadata.logical_to_all_physical_map_cpu
    identity = torch.arange(n).expand(metadata.num_layers, n)
    return (metadata.ep_size == 1 and metadata.num_physical_experts == n
            and torch.equal(p2l, identity) and torch.equal(l2p[:, :, 0], identity)
            and bool((l2p[:, :, 1:] == -1).all()))


def guard_draft_expert_map():
    """Let the full-512 drafter load against the 448-expert target's expert map.

    SGLang keeps one process-wide expert map, sized by the target. The drafter's
    MoE weight loader indexes it with drafter expert ids 448..511 (IndexError), and
    for global-expert params the stock range() would silently return no slots.
    On one rank with the trivial layout every map query is the identity, so ids
    past the target's experts get the identity too; anything else refuses.
    """
    from sglang.srt.eplb.expert_location import ExpertLocationMetadata
    original = ExpertLocationMetadata.logical_to_all_physical
    if getattr(original, 'retained_pruning', False):
        return

    def logical_to_all_physical(self, layer_id, logical_expert_id, require_global_experts=False):
        if logical_expert_id < self.num_logical_experts:
            return original(self, layer_id, logical_expert_id, require_global_experts)
        if not 0 <= logical_expert_id < DRAFT_EXPERTS:
            raise ValueError(f'Expert id {logical_expert_id} outside the drafter bank')
        if not is_identity_layout(self):
            raise ValueError('Expert map is not the one-rank identity; drafter experts cannot be placed')
        return [logical_expert_id]

    logical_to_all_physical.retained_pruning = True
    ExpertLocationMetadata.logical_to_all_physical = logical_to_all_physical
    print('RETAINED_PRUNING_DRAFT_MAP_GUARD installed', flush=True)


def record_draft(model):
    directory = os.environ.get('RETAINED_PRUNING_RECEIPT_DIR')
    if not directory:
        return
    shapes = expert_shapes(model)
    if model.config.num_experts != 512 or not shapes or any(s[0] != 512 for s in shapes.values()):
        raise ValueError('Drafter expert bank was changed')
    guard_draft_expert_map()
    output = Path(directory) / 'draft.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'draft_experts': 512, 'shapes': shapes}, indent=2) + '\n')
    print('RETAINED_PRUNING_DRAFT experts=512', flush=True)


def make_view(source, destination, mask, router_report):
    source, destination = Path(source), Path(destination)
    config = json.loads((source / 'config.json').read_text())
    if set(mask) != {str(i) for i in range(48)} or any(len(v) != 448 or len(set(v)) != 448 or any(type(x) is not int or not 0 <= x < 512 for x in v) for v in mask.values()):
        raise ValueError('Mask is not 48 valid 448-of-512 selections')
    text = config['text_config']
    if text['num_experts'] != 512 or text['num_experts_per_tok'] != 10 or text['num_hidden_layers'] != 48:
        raise ValueError('Unexpected source architecture')
    if any('.mlp.experts.' in name and not name.startswith('mtp.') for name in config['quantization_config']['extra_config']):
        raise ValueError('Per-expert quantization overrides need explicit remapping')
    if not router_report['all_equal'] or len(router_report['routers']) != 48:
        raise ValueError('Router transfer not qualified')
    hashes = {r['name'].split('.layers.')[1].split('.')[0]: r['sha256']['intel'] for r in router_report['routers']}
    if set(hashes) != set(mask):
        raise ValueError('Router evidence incomplete')
    text['num_experts'] = 448
    text['retained_pruning'] = {'mask': mask, 'router_hashes': hashes,
                                'mask_sha256': hashlib.sha256(json.dumps(mask, sort_keys=True).encode()).hexdigest()}
    destination.mkdir(parents=True, exist_ok=False)
    for path in source.iterdir():
        if path.name != 'config.json':
            (destination / path.name).symlink_to(path.resolve())
    (destination / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    return destination
