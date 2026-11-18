"""Offline qualification of flashnext_prune: real RadixArk names, real routers, stub vLLM classes.

  ~/projects/arc-agi-3-reference/duck-critic-adapter/ARC3-Inference/.venv/bin/python -B prune/test_plugin.py
"""
import json, os, sys, types
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "plugin")); sys.path.insert(0, str(HERE))
import flashnext_prune as fp
from recover_mask import REPOS, tensor

mask, keep = fp.load_mask(HERE / "k448_mask_layers.json")
assert keep == 448 and len(mask) == 48

# 1. filter over every real checkpoint name (full HF names and vLLM-relative names)
wm = json.load(open(REPOS["rx"][2]))["weight_map"]
for strip in ("", "model.language_model."):
    names = [n[len(strip):] if n.startswith(strip) else n for n in wm]
    fake = lambda n: np.zeros((512, 2)) if n.endswith("mlp.gate.weight") else np.zeros(1)
    stats = {"dropped": 0, "renumbered": 0, "routers": 0}
    out = list(fp.filter_weights(((n, fake(n)) for n in names), mask, stats))
    assert stats == {"dropped": 64 * 48 * 12, "renumbered": 448 * 48 * 12, "routers": 48}, stats
    got = {n for n, _ in out}
    assert not any(f".experts.{e}." in n for n in got if "mtp." not in n for e in range(448, 512)), "expert id >= 448 survived"
    mtp_in = {n for n in names if "mtp." in n}
    assert mtp_in <= got, "MTP tensors changed"
    assert all(t.shape == (448, 2) for n, t in out if n.endswith("mlp.gate.weight") and "mtp." not in n)
    print("filter ok", strip or "(full names)", stats, len(got))

# 2. slicing KF's real routers with the mask reproduces k448's routers byte for byte
kwm = json.load(open(REPOS["k448"][2]))["weight_map"]
for layer in (0, 15, 47):
    name = f"model.language_model.layers.{layer}.mlp.gate.weight"
    _, rx = tensor(REPOS["rx"][0], REPOS["rx"][1], wm[name], name)
    _, k4 = tensor(REPOS["k448"][0], REPOS["k448"][1], kwm[name], name)
    sliced = fp._take_rows(rx, mask[layer])
    assert sliced.tobytes() == k4.tobytes(), layer
print("router slice == k448 router for layers 0,15,47")

# 3. stub vLLM: target MoE blocks see 448, PLE gate and MTP draft see 512, load fuses shared at 448
seen = []
class Cfg:
    def __init__(self): self.num_experts = 512
class Moe:
    def __init__(self, vllm_config, prefix=""):
        seen.append((prefix, vllm_config.model_config.hf_text_config.num_experts))
def vcfg(c): return types.SimpleNamespace(model_config=types.SimpleNamespace(hf_text_config=c))
class Top:
    def __init__(self, model): self.model = model
    def load_weights(self, weights):
        # like AutoWeightsLoader._groupby_prefix: one Model.load_weights call per contiguous shard run
        import itertools
        for _, grp in itertools.groupby(weights, key=lambda nw: wm[nw[0]]):
            self.model.load_weights(grp)
class Model:
    def __init__(self, *, vllm_config, prefix=""):
        self.config = vllm_config.model_config.hf_text_config
        seen.append(("ple-gate", self.config.num_experts))
        self.blocks = [Moe(vllm_config, prefix=f"layers.{i}.mlp") for i in range(48)]
    def load_weights(self, weights):
        n = self.config.num_experts; names = [w for w, _ in weights]
        seen.append(("fuse-at", n)); return set(names)
mod = types.ModuleType("m"); mod.Qwen3_8FlashNextModel = Model; mod.Qwen3_8FlashNextSparseMoeBlock = Moe; mod.Qwen3_8FlashNextForConditionalGeneration = Top
for name in ("vllm", "vllm.models", "vllm.models.qwen3_8_flash_next", "vllm.models.qwen3_8_flash_next.nvidia"):
    sys.modules[name] = types.ModuleType(name)
sys.modules["vllm.models.qwen3_8_flash_next.nvidia"].model = mod
sys.modules["vllm.models.qwen3_8_flash_next.nvidia.model"] = mod
os.environ["FLASHNEXT_PRUNE_MASK"] = str(HERE / "k448_mask_layers.json")
fp.register(); fp.register()                           # idempotent
target_cfg, draft_cfg = Cfg(), Cfg()
model = Model(vllm_config=vcfg(target_cfg))
Moe(vcfg(draft_cfg), prefix="mtp.layers.0.mlp")        # MTP draft built outside the target init
names = [n for n in wm]
Top(model).load_weights(((n, np.zeros((512, 2)) if n.endswith("mlp.gate.weight") else np.zeros(1)) for n in sorted(names, key=lambda n: wm[n])))
assert seen[0] == ("ple-gate", 512), seen[0]
assert all(v == 448 for p, v in seen[1:49]), "target MoE not 448"
assert seen[49] == ("mtp.layers.0.mlp", 512), seen[49]
assert all(x == ("fuse-at", 448) for x in seen[50:]) and len(seen) > 51, seen[50:53]
assert target_cfg.num_experts == 512 and draft_cfg.num_experts == 512, "config not restored"
print(f"multi-call load: {len(seen) - 50} Model.load_weights calls;"); print("stub wiring ok: ple-gate 512, 48 target blocks 448, MTP 512, fuse at 448, configs restored")
