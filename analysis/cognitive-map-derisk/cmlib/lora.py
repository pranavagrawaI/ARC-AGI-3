import math

import torch
import torch.nn as nn

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.base = base
        self.scale = alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features, dtype=torch.float32, device=base.weight.device))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32, device=base.weight.device))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + ((x.float() @ self.A.T) @ self.B.T * self.scale).to(x.dtype)


def apply_lora(model, r=8, alpha=16):
    for p in model.parameters():
        p.requires_grad_(False)
    n = 0
    for name, mod in list(model.named_modules()):
        for t in TARGETS:
            if hasattr(mod, t) and isinstance(getattr(mod, t), nn.Linear):
                setattr(mod, t, LoRALinear(getattr(mod, t), r, alpha))
                n += 1
    return n


def lora_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if k.endswith(".A") or k.endswith(".B")}


def load_lora_state(model, sd):
    missing = model.load_state_dict(sd, strict=False)
    assert not missing.unexpected_keys, missing.unexpected_keys
