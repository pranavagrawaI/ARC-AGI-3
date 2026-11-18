"""Prune Qwen3.8-Flash-Next's routed experts at load time with a fixed per-layer keep mask (REAP-k448).

A vLLM general plugin (entry point `vllm.general_plugins`), so it runs in every engine/worker process.
Target model only: MoE blocks built inside `Qwen3_8FlashNextModel.__init__` see `num_experts = len(keep)`;
everything else (the PLE identity gate, the MTP draft, which builds from its own draft config) keeps seeing
the checkpoint's 512. `Qwen3_8FlashNextModel.load_weights` drops pruned experts, renumbers kept experts
0..K-1 and slices router rows, then fuses the shared expert at index K.

Env: FLASHNEXT_PRUNE_MASK=/path/mask.json  ({"<layer>": [kept original expert ids, in new order]}).
Unset -> the plugin does nothing.
"""
import contextlib
import functools
import json
import os
import re
import sys
import threading

_EXPERT = re.compile(r"(^|\.)layers\.(\d+)\.mlp\.experts\.(\d+)\.")
_ROUTER = re.compile(r"(^|\.)layers\.(\d+)\.mlp\.gate\.weight$")
_STATE = threading.local()


def _log(msg):
    print(f"FLASHNEXT_PRUNE pid={os.getpid()} {msg}", file=sys.stderr, flush=True)


def load_mask(path):
    raw = json.loads(open(path).read())
    mask = {int(k): list(map(int, v)) for k, v in raw.items()}
    sizes = {len(v) for v in mask.values()}
    assert len(sizes) == 1, f"ragged mask {sizes}"
    for layer, kept in mask.items():
        assert len(set(kept)) == len(kept), f"duplicate expert in layer {layer}"
    return mask, sizes.pop()


def filter_weights(weights, mask, stats):
    """Yield (name, tensor) with pruned experts dropped, kept experts renumbered, routers row-sliced.
    Names outside `mask` layers (e.g. MTP) pass through unchanged."""
    remap = {layer: {old: new for new, old in enumerate(kept)} for layer, kept in mask.items()}
    for name, tensor in weights:
        if "mtp." in name:          # the MTP bank keeps all experts
            yield name, tensor
            continue
        m = _EXPERT.search(name)
        if m and int(m[2]) in remap:
            layer, old = int(m[2]), int(m[3])
            new = remap[layer].get(old)
            if new is None:
                stats["dropped"] += 1
                continue
            stats["renumbered"] += 1
            yield name[:m.start(3)] + str(new) + name[m.end(3):], tensor
            continue
        r = _ROUTER.search(name)
        if r and int(r[2]) in remap:
            kept = mask[int(r[2])]
            assert tensor.shape[0] > max(kept), (name, tuple(tensor.shape))
            stats["routers"] += 1
            yield name, _take_rows(tensor, kept)
            continue
        yield name, tensor


def _take_rows(tensor, rows):
    try:
        import torch
        return tensor.index_select(0, torch.as_tensor(rows, dtype=torch.long, device=tensor.device)).contiguous()
    except ImportError:          # offline tests with numpy arrays
        return tensor[rows]


@contextlib.contextmanager
def _num_experts(config, value):
    saved = config.num_experts
    config.num_experts = value
    try:
        yield
    finally:
        config.num_experts = saved


def register():
    path = os.environ.get("FLASHNEXT_PRUNE_MASK", "").strip()
    if not path:
        return
    try:
        from vllm.models.qwen3_8_flash_next.nvidia import model as m
    except Exception as exc:        # API-server-only processes without the model stack
        _log(f"skip: {type(exc).__name__}: {exc}")
        return
    if getattr(m, "_flashnext_prune", False):
        return
    mask, keep = load_mask(path)
    Model, Moe = m.Qwen3_8FlashNextModel, m.Qwen3_8FlashNextSparseMoeBlock
    orig_model_init, orig_moe_init, orig_load = Model.__init__, Moe.__init__, Model.load_weights

    def model_init(self, *args, **kwargs):
        _STATE.target = True        # MoE blocks built during THIS init are target-model blocks
        try:
            return orig_model_init(self, *args, **kwargs)
        finally:
            _STATE.target = False

    def moe_init(self, vllm_config, prefix=""):
        cfg = vllm_config.model_config.hf_text_config
        if getattr(_STATE, "target", False):
            assert cfg.num_experts == 512, cfg.num_experts
            with _num_experts(cfg, keep):
                out = orig_moe_init(self, vllm_config, prefix=prefix)
            _STATE.blocks = getattr(_STATE, "blocks", 0) + 1
            return out
        return orig_moe_init(self, vllm_config, prefix=prefix)

    totals = {"dropped": 0, "renumbered": 0, "routers": 0, "calls": 0}

    def load_weights(self, weights):
        # AutoWeightsLoader groups the checkpoint stream by submodule, so this runs once per contiguous run of
        # text-model tensors (routers sit in separate BF16 shards). Count across calls; check once at the top.
        stats = {"dropped": 0, "renumbered": 0, "routers": 0}
        with _num_experts(self.config, keep):     # maybe_fuse_shared_experts puts the shared expert at index keep
            loaded = orig_load(self, filter_weights(weights, mask, stats))
        for k, v in stats.items():
            totals[k] += v
        totals["calls"] += 1
        return loaded

    def check_totals():
        layers = len(mask)
        _log(f"load_weights totals {totals} blocks={getattr(_STATE, 'blocks', 0)} keep={keep} layers={layers}")
        per_expert = totals["renumbered"] // (keep * layers)
        assert per_expert > 0 and totals["routers"] == layers, totals
        assert totals["renumbered"] == per_expert * keep * layers, totals
        assert totals["dropped"] == per_expert * (512 - keep) * layers, totals

    tops = [getattr(m, n) for n in ("Qwen3_8FlashNextForConditionalGeneration", "Qwen3_8FlashNextForCausalLM") if hasattr(m, n)]

    def wrap_top(cls):
        orig = cls.load_weights

        def top_load_weights(self, weights):
            depth = getattr(_STATE, "top_depth", 0)
            _STATE.top_depth = depth + 1
            try:
                loaded = orig(self, weights)
            finally:
                _STATE.top_depth = depth
            if depth == 0:                      # outermost model only (CausalLM may nest inside ConditionalGeneration)
                check_totals()
            return loaded
        cls.load_weights = functools.wraps(orig)(top_load_weights)

    Model.__init__ = functools.wraps(orig_model_init)(model_init)
    Moe.__init__ = functools.wraps(orig_moe_init)(moe_init)
    Model.load_weights = functools.wraps(orig_load)(load_weights)
    for cls in tops:
        wrap_top(cls)
    m._flashnext_prune = True
    _log(f"registered keep={keep} layers={len(mask)} mask={path}")
